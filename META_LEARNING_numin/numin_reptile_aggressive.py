"""
Aggressive Reptile for Numin2 - Maximum performance configuration
"""

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NuminDataset:
    def __init__(self, parquet_path, window_size=50, support_days=5):
        self.window_size = window_size
        self.support_days = support_days
        self.tasks = []
        self.task_keys = []  # (year, month) for chronological sorting

        df = pd.read_parquet(parquet_path)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df['year'] = df.index.year
        df['month'] = df.index.month

        for (year, month), group in df.groupby(['year', 'month']):
            group = group.sort_index()
            returns = group.drop(['year', 'month'], axis=1).values

            if len(returns) < window_size + support_days + 1:
                continue

            samples, targets = [], []
            for i in range(window_size, len(returns)):
                window = returns[i - window_size:i]
                ranks = np.argsort(np.argsort(-returns[i]))
                samples.append(window)
                targets.append(ranks)

            if len(samples) >= support_days + 1:
                self.tasks.append({
                    'samples': np.array(samples, dtype=np.float32),
                    'targets': np.array(targets, dtype=np.int64)
                })
                self.task_keys.append((year, month))

        # Sort tasks chronologically by (year, month)
        sorted_indices = sorted(range(len(self.task_keys)), key=lambda i: self.task_keys[i])
        self.tasks = [self.tasks[i] for i in sorted_indices]
        self.task_keys = [self.task_keys[i] for i in sorted_indices]

        print(f"Created {len(self.tasks)} tasks")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        samples = task['samples'].copy()
        targets = task['targets'].copy()

        support_idx = list(range(min(self.support_days, len(samples)-1)))
        query_idx = list(range(len(support_idx), len(samples)))

        # Normalize using support-only statistics
        support_data = samples[support_idx]
        mean, std = support_data.mean(), support_data.std() + 1e-8
        samples = (samples - mean) / std

        return {
            'support_samples': torch.tensor(samples[support_idx]),
            'support_targets': torch.tensor(targets[support_idx]),
            'query_samples': torch.tensor(samples[query_idx]),
            'query_targets': torch.tensor(targets[query_idx]),
        }

class LargeEncoder(nn.Module):
    """Larger encoder with more capacity"""
    def __init__(self, input_dim=50, hidden_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = self.norm(out.mean(dim=1))
        return self.proj(out)

class NuminModel(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=512):
        super().__init__()
        self.encoder = LargeEncoder(num_stocks, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stocks * num_stocks)
        )
        self.num_stocks = num_stocks

    def forward(self, x):
        enc = self.encoder(x)
        logits = self.decoder(enc)
        return logits.view(-1, self.num_stocks, self.num_stocks)

class Reptile:
    def __init__(self, model, outer_lr=0.1, inner_lr=0.01, inner_steps=20, device='cuda'):
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

        support_samples = task['support_samples'].to(self.device)
        support_targets = task['support_targets'].to(self.device)
        query_samples = task['query_samples'].to(self.device)
        query_targets = task['query_targets'].to(self.device)

        old_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr, momentum=0.9)

        for _ in range(self.inner_steps):
            logits = self.model(support_samples)
            loss = self.compute_loss(logits, support_targets)
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            inner_opt.step()

        # Evaluate on query BEFORE Reptile interpolation (model has adapted weights)
        with torch.no_grad():
            query_logits = self.model(query_samples)
            query_loss = self.compute_loss(query_logits, query_targets)
            acc = (query_logits.argmax(dim=-1) == query_targets).float().mean().item()

        # Reptile update: move towards trained weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = old_weights[name] + self.outer_lr * (param.data - old_weights[name])

        return {'loss': query_loss.item(), 'accuracy': acc}

    def evaluate(self, dataset, indices):
        all_corrs = []

        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]

            support_samples = task['support_samples'].to(self.device)
            support_targets = task['support_targets'].to(self.device)
            query_samples = task['query_samples'].to(self.device)
            query_targets = task['query_targets'].to(self.device)

            old_weights = {name: param.clone() for name, param in self.model.named_parameters()}
            inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr, momentum=0.9)
            self.model.train()

            for _ in range(self.inner_steps):
                logits = self.model(support_samples)
                loss = self.compute_loss(logits, support_targets)
                inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                inner_opt.step()

            self.model.eval()
            with torch.no_grad():
                query_logits = self.model(query_samples)
                preds = query_logits.argmax(dim=-1).cpu().numpy()
                tgts = query_targets.cpu().numpy()

                for i in range(len(preds)):
                    corr, _ = spearmanr(preds[i], tgts[i])
                    if not np.isnan(corr):
                        all_corrs.append(corr)

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = old_weights[name]

        return {'mean_correlation': np.mean(all_corrs) if all_corrs else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='numin_sample.parquet')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.15)
    parser.add_argument('--inner_steps', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_numin_aggressive')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = NuminDataset(args.data_path, window_size=args.window_size)

    # Temporal (chronological) split -- no shuffle
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, n))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = NuminModel(num_stocks=50, hidden_dim=args.hidden_dim)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    reptile = Reptile(model, args.outer_lr, args.inner_lr, args.inner_steps, device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_corr = -1

    for epoch in range(args.epochs):
        # Outer lr / epsilon decay
        reptile.outer_lr = args.outer_lr * (1.0 - epoch / args.epochs)

        random.shuffle(train_idx)
        losses, accs = [], []

        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}"):
            metrics = reptile.train_step(dataset[idx])
            losses.append(metrics['loss'])
            accs.append(metrics['accuracy'])

        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Acc={np.mean(accs):.4f}")

        if (epoch + 1) % 5 == 0:
            val_results = reptile.evaluate(dataset, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_correlation'] > best_val_corr:
                best_val_corr = val_results['mean_correlation']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
                print(f"New best: {best_val_corr:.4f}")

    # Reload best model before test
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model for testing.")

    test_results = reptile.evaluate(dataset, test_idx)
    print(f"\nTest: {test_results}")

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_corr': best_val_corr,
            'test_correlation': test_results['mean_correlation'],
            'args': vars(args)
        }, f, indent=2)

    print(f"\nBest Val Corr: {best_val_corr:.4f}")
    print(f"Test Corr: {test_results['mean_correlation']:.4f}")

if __name__ == '__main__':
    main()
