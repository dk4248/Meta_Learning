"""
Ensemble model for Numin2 - Combines Reptile + ProtoNet predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import argparse
import json
import os
from scipy.stats import spearmanr

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
                    'targets': np.array(targets, dtype=np.int64),
                    'key': (year, month),
                })

        # Sort tasks chronologically
        self.tasks.sort(key=lambda t: t['key'])
        print(f"Created {len(self.tasks)} tasks")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        samples = task['samples'].copy()
        targets = task['targets']

        support_idx = list(range(min(self.support_days, len(samples)-1)))
        query_idx = list(range(len(support_idx), len(samples)))

        # Support-only normalization
        support = samples[support_idx]
        mean, std = support.mean(), support.std() + 1e-8
        samples = (samples - mean) / std

        return {
            'support_samples': torch.tensor(samples[support_idx]),
            'support_targets': torch.tensor(targets[support_idx]),
            'query_samples': torch.tensor(samples[query_idx]),
            'query_targets': torch.tensor(targets[query_idx]),
        }

# Model 1: LSTM-based (like Reptile)
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.norm(out.mean(dim=1))

class Model1(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=256):
        super().__init__()
        self.encoder = LSTMEncoder(num_stocks, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stocks * num_stocks)
        )
        self.num_stocks = num_stocks

    def forward(self, x):
        enc = self.encoder(x)
        logits = self.decoder(enc)
        return logits.view(-1, self.num_stocks, self.num_stocks)

# Model 2: Transformer-based
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        return self.norm(x.mean(dim=1))

class Model2(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=256):
        super().__init__()
        self.encoder = TransformerEncoder(num_stocks, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stocks * num_stocks)
        )
        self.num_stocks = num_stocks

    def forward(self, x):
        enc = self.encoder(x)
        logits = self.decoder(enc)
        return logits.view(-1, self.num_stocks, self.num_stocks)

class EnsembleModel(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=256):
        super().__init__()
        self.model1 = Model1(num_stocks, hidden_dim)
        self.model2 = Model2(num_stocks, hidden_dim)
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.num_stocks = num_stocks

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)
        w = F.softmax(self.weight, dim=0)
        return w[0] * logits1 + w[1] * logits2

class EnsembleReptile:
    def __init__(self, model, outer_lr=0.1, inner_lr=0.01, inner_steps=15, device='cuda'):
        self.model = model.to(device)
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device

    def compute_loss(self, logits, targets):
        ns = self.model.num_stocks
        return F.cross_entropy(logits.view(-1, ns), targets.view(-1))

    def train_step(self, task):
        self.model.train()

        support_samples = task['support_samples'].to(self.device)
        support_targets = task['support_targets'].to(self.device)
        query_samples = task['query_samples'].to(self.device)
        query_targets = task['query_targets'].to(self.device)

        old_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            logits = self.model(support_samples)
            loss = self.compute_loss(logits, support_targets)
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            inner_opt.step()

        # Evaluate BEFORE Reptile interpolation (to measure adapted model quality)
        self.model.eval()
        with torch.no_grad():
            query_logits = self.model(query_samples)
            acc = (query_logits.argmax(dim=-1) == query_targets).float().mean().item()

        # Reptile interpolation
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = old_weights[name] + self.outer_lr * (param.data - old_weights[name])

        return {'accuracy': acc}

    def evaluate(self, dataset, indices):
        all_corrs = []

        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]

            support_samples = task['support_samples'].to(self.device)
            support_targets = task['support_targets'].to(self.device)
            query_samples = task['query_samples'].to(self.device)
            query_targets = task['query_targets'].to(self.device)

            old_weights = {name: param.clone() for name, param in self.model.named_parameters()}
            inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            self.model.train()

            for _ in range(self.inner_steps):
                logits = self.model(support_samples)
                loss = self.compute_loss(logits, support_targets)
                inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
    parser.add_argument('--outer_lr', type=float, default=0.12)
    parser.add_argument('--inner_steps', type=int, default=15)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_numin_ensemble')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = NuminDataset(args.data_path)

    # Temporal split (tasks already sorted chronologically)
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, n))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = EnsembleModel(num_stocks=50, hidden_dim=args.hidden_dim)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    reptile = EnsembleReptile(model, args.outer_lr, args.inner_lr, args.inner_steps, device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_corr = -1

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        accs = []

        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}"):
            metrics = reptile.train_step(dataset[idx])
            accs.append(metrics['accuracy'])

        print(f"Epoch {epoch+1}: Acc={np.mean(accs):.4f}")

        # Decay outer_lr
        reptile.outer_lr = args.outer_lr * (1 - epoch / args.epochs)

        if (epoch + 1) % 5 == 0:
            val_results = reptile.evaluate(dataset, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_correlation'] > best_val_corr:
                best_val_corr = val_results['mean_correlation']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

    # Reload best model before test
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

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
