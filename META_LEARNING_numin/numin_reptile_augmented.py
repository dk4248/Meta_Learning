"""Reptile with Task Augmentation for Numin2 - data augmentation at the task level."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import random
import argparse
import json
import os

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

class NuminAugDataset:
    def __init__(self, path, window_size=50, support_days=5):
        self.window_size = window_size
        self.support_days = support_days
        self.tasks = []
        self.task_keys = []  # (year, month) for chronological sorting

        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df['year'] = df.index.year
        df['month'] = df.index.month

        for (year, month), g in df.groupby(['year', 'month']):
            g = g.sort_index()
            r = g.drop(['year', 'month'], axis=1).values
            if len(r) < window_size + support_days + 1:
                continue
            S, T = [], []
            for i in range(window_size, len(r)):
                S.append(r[i - window_size:i])
                T.append(np.argsort(np.argsort(-r[i])))
            if len(S) >= support_days + 1:
                self.tasks.append({
                    'samples': np.array(S, dtype=np.float32),
                    'targets': np.array(T, dtype=np.int64)
                })
                self.task_keys.append((year, month))

        # Sort tasks chronologically by (year, month)
        sorted_indices = sorted(range(len(self.task_keys)), key=lambda i: self.task_keys[i])
        self.tasks = [self.tasks[i] for i in sorted_indices]
        self.task_keys = [self.task_keys[i] for i in sorted_indices]

        print(f"Created {len(self.tasks)} base tasks")

    def __len__(self):
        return len(self.tasks)

    def augment(self, samples, targets):
        s = samples.copy()
        t = targets.copy()
        # 1. Gaussian noise
        if random.random() < 0.5:
            s = s + np.random.randn(*s.shape).astype(np.float32) * 0.01
        # 2. Scale augmentation
        if random.random() < 0.5:
            s = s * np.random.uniform(0.8, 1.2)
        # 3. Time reversal
        if random.random() < 0.2:
            s = s[::-1].copy()
            t = t[::-1].copy()
        # 4. Stock subsampling (mask 10 random stocks)
        if random.random() < 0.3:
            mask_idx = random.sample(range(50), 10)
            for mi in mask_idx:
                s[:, :, mi] = 0
        return s.astype(np.float32), t

    def __getitem__(self, idx):
        task = self.tasks[idx]
        s = task['samples'].copy()
        tg = task['targets'].copy()
        s, tg = self.augment(s, tg)

        si = list(range(min(self.support_days, len(s) - 1)))
        qi = list(range(len(si), len(s)))

        # Normalize using support-only statistics
        support_data = s[si]
        m, st = support_data.mean(), support_data.std() + 1e-8
        s = (s - m) / st

        return {
            'support_samples': torch.tensor(s[si]),
            'support_targets': torch.tensor(tg[si]),
            'query_samples': torch.tensor(s[qi]),
            'query_targets': torch.tensor(tg[qi]),
        }

class Model(nn.Module):
    def __init__(self, ns=50, hd=256):
        super().__init__()
        self.lstm = nn.LSTM(ns, hd, 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hd * 2)
        self.dec = nn.Sequential(
            nn.Linear(hd * 2, hd),
            nn.ReLU(),
            nn.LayerNorm(hd),
            nn.Linear(hd, ns * ns)
        )
        self.ns = ns

    def forward(self, x):
        self.lstm.flatten_parameters()
        o, _ = self.lstm(x)
        o = self.norm(o.mean(1))
        return self.dec(o).view(-1, self.ns, self.ns)

class Reptile:
    def __init__(self, model, outer_lr=0.1, inner_lr=0.01, inner_steps=10, device='cuda'):
        self.model = model.to(device)
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device

    def compute_loss(self, logits, targets):
        B, S, C = logits.shape
        return F.cross_entropy(logits.reshape(B * S, C), targets.reshape(B * S))

    def train_step(self, task):
        self.model.train()
        ss = task['support_samples'].to(self.device)
        st = task['support_targets'].to(self.device)
        qs = task['query_samples'].to(self.device)
        qt = task['query_targets'].to(self.device)

        ow = {n: p.clone() for n, p in self.model.named_parameters()}
        iopt = optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            l = self.compute_loss(self.model(ss), st)
            iopt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            iopt.step()

        # Evaluate on query BEFORE Reptile interpolation (model has adapted weights)
        with torch.no_grad():
            query_logits = self.model(qs)
            query_loss = self.compute_loss(query_logits, qt)
            acc = (query_logits.argmax(dim=-1) == qt).float().mean().item()

        # Reptile update: move towards trained weights
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.data = ow[n] + self.outer_lr * (p.data - ow[n])

        return {'loss': query_loss.item(), 'accuracy': acc}

    def evaluate(self, ds, indices):
        corrs = []
        for i in tqdm(indices, desc="Eval"):
            task = ds[i]
            ss = task['support_samples'].to(self.device)
            st = task['support_targets'].to(self.device)
            qs = task['query_samples'].to(self.device)
            qt = task['query_targets'].to(self.device)

            ow = {n: p.clone() for n, p in self.model.named_parameters()}
            iopt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            self.model.train()

            for _ in range(self.inner_steps):
                l = self.compute_loss(self.model(ss), st)
                iopt.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                iopt.step()

            self.model.eval()
            with torch.no_grad():
                pred = self.model(qs).argmax(-1).cpu().numpy()
                tgt = qt.cpu().numpy()
                for j in range(len(pred)):
                    c, _ = spearmanr(pred[j], tgt[j])
                    if not np.isnan(c):
                        corrs.append(c)

            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    p.data = ow[n]

        return {'mean_correlation': np.mean(corrs) if corrs else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='numin_sample.parquet')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.1)
    parser.add_argument('--inner_steps', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='checkpoints_numin_augmented')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    ds = NuminAugDataset(args.data_path)

    # Temporal (chronological) split -- no shuffle
    n = len(ds)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    tr = list(range(train_size))
    va = list(range(train_size, train_size + val_size))
    te = list(range(train_size + val_size, n))

    model = Model(50, args.hidden_dim)
    reptile = Reptile(model, args.outer_lr, args.inner_lr, args.inner_steps, device)
    os.makedirs(args.save_dir, exist_ok=True)
    best_val = -1

    # Use non-augmented dataset for eval
    eval_ds = NuminAugDataset.__new__(NuminAugDataset)
    eval_ds.tasks = ds.tasks
    eval_ds.window_size = ds.window_size
    eval_ds.support_days = ds.support_days
    eval_ds.task_keys = ds.task_keys
    eval_ds.augment = lambda s, t: (s, t)  # no augmentation for eval

    for ep in range(args.epochs):
        # Outer lr / epsilon decay
        reptile.outer_lr = args.outer_lr * (1.0 - ep / args.epochs)

        random.shuffle(tr)
        losses, accs = [], []

        for i in tqdm(tr, desc=f"Epoch {ep+1}"):
            metrics = reptile.train_step(ds[i])
            losses.append(metrics['loss'])
            accs.append(metrics['accuracy'])

        print(f"Epoch {ep+1}: Loss={np.mean(losses):.4f}, Acc={np.mean(accs):.4f}")

        if (ep + 1) % 5 == 0:
            r = reptile.evaluate(eval_ds, va)
            print(f"Epoch {ep+1} Val: {r}")
            if r['mean_correlation'] > best_val:
                best_val = r['mean_correlation']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

    # Reload best model before test
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model for testing.")

    tr_ = reptile.evaluate(eval_ds, te)
    print(f"Test: {tr_}")

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_corr': best_val,
            'test_correlation': tr_['mean_correlation'],
            'args': vars(args)
        }, f, indent=2)

if __name__ == '__main__':
    main()
