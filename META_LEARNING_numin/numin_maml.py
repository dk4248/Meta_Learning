"""
Numin2 Meta-Learning with MAML

MAML for financial time series prediction on the Numin2 dataset.
Each month is a separate "task". The model meta-learns an initialization
that can quickly adapt to new market conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.func import functional_call
import numpy as np
from tqdm import tqdm
import random
import argparse
import json
import os
from scipy.stats import spearmanr

# Disable efficient SDPA for double backward compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NuminLocalDataset(Dataset):
    """Local dataset for Numin2. Each task is a month of trading data."""

    def __init__(self, parquet_path, window_size=50, support_days=5, device='cuda'):
        import pandas as pd
        self.window_size = window_size
        self.support_days = support_days
        self.device = device
        self.tasks = []

        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df['year'] = df.index.year
        df['month'] = df.index.month
        grouped = df.groupby(['year', 'month'])

        for (year, month), group in tqdm(grouped, desc="Processing months"):
            group = group.sort_index()
            returns = group.drop(['year', 'month'], axis=1).values

            if len(returns) < window_size + support_days + 1:
                continue

            samples = []
            targets = []
            for i in range(window_size, len(returns)):
                window = returns[i - window_size:i]
                next_returns = returns[i]
                ranks = np.argsort(np.argsort(-next_returns))
                samples.append(window)
                targets.append(ranks)

            if len(samples) >= support_days + 1:
                self.tasks.append({
                    'year': int(year), 'month': int(month),
                    'samples': np.array(samples, dtype=np.float32),
                    'targets': np.array(targets, dtype=np.int64)
                })

        # Sort by (year, month) for chronological splitting
        self.tasks.sort(key=lambda t: (t['year'], t['month']))
        print(f"Created {len(self.tasks)} tasks (months)")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        samples = task['samples'].copy()
        targets = task['targets']

        num_samples = len(samples)
        support_size = min(self.support_days, num_samples - 1)
        support_idx = list(range(support_size))
        query_idx = list(range(support_size, num_samples))

        if len(query_idx) == 0:
            query_idx = [num_samples - 1]
            support_idx = support_idx[:-1]

        # Normalize using SUPPORT statistics only (no data leakage)
        support_data = samples[support_idx]
        mean = support_data.mean()
        std = support_data.std() + 1e-8
        samples = (samples - mean) / std

        return {
            'support_samples': torch.tensor(samples[support_idx], dtype=torch.float32),
            'support_targets': torch.tensor(targets[support_idx], dtype=torch.long),
            'query_samples': torch.tensor(samples[query_idx], dtype=torch.float32),
            'query_targets': torch.tensor(targets[query_idx], dtype=torch.long),
        }


class FinancialEncoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        return out.mean(dim=1)


class NuminRankModel(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_stocks = num_stocks
        self.encoder = FinancialEncoder(num_stocks, hidden_dim, num_layers, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stocks * num_stocks)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.decoder(encoded)
        return logits.view(-1, self.num_stocks, self.num_stocks)


class NuminMAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, device='cuda'):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def compute_rank_loss(self, logits, targets):
        B, S, C = logits.shape
        return F.cross_entropy(logits.reshape(B * S, C), targets.reshape(B * S))

    def compute_correlation(self, logits, targets):
        # Use expected rank (soft prediction) instead of argmax
        rank_indices = torch.arange(logits.size(-1), device=logits.device).float()
        predictions = (logits.softmax(dim=-1) * rank_indices).sum(dim=-1)

        correlations = []
        for b in range(predictions.size(0)):
            pred = predictions[b].cpu().numpy()
            tgt = targets[b].cpu().numpy()
            corr, _ = spearmanr(pred, tgt)
            if not np.isnan(corr):
                correlations.append(corr)
        return np.mean(correlations) if correlations else 0

    def train_step(self, task):
        self.model.train()
        ss = task['support_samples'].to(self.device)
        st = task['support_targets'].to(self.device)
        qs = task['query_samples'].to(self.device)
        qt = task['query_targets'].to(self.device)

        # Proper MAML: use functional_call with differentiable inner loop
        # Disable cuDNN for LSTM (cuDNN doesn't support double backward)
        fast_params = {k: v.clone() for k, v in dict(self.model.named_parameters()).items()}

        with torch.backends.cudnn.flags(enabled=False):
            for _ in range(self.inner_steps):
                logits = functional_call(self.model, fast_params, (ss,))
                loss = self.compute_rank_loss(logits, st)
                grads = torch.autograd.grad(loss, fast_params.values(), create_graph=True)
                fast_params = {k: v - self.inner_lr * g
                              for (k, v), g in zip(fast_params.items(), grads)}

            # Query loss with adapted params - gradients flow to model.parameters()
            query_logits = functional_call(self.model, fast_params, (qs,))
            query_loss = self.compute_rank_loss(query_logits, qt)

        self.optimizer.zero_grad()
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        with torch.no_grad():
            acc = (query_logits.argmax(-1) == qt).float().mean().item()
        return {'loss': query_loss.item(), 'accuracy': acc}

    def evaluate(self, dataset, indices=None):
        self.model.train()  # Keep train mode for LSTM backward compatibility
        if indices is None:
            indices = list(range(len(dataset)))

        all_losses, all_accs, all_corrs = [], [], []

        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]
            ss = task['support_samples'].to(self.device)
            st = task['support_targets'].to(self.device)
            qs = task['query_samples'].to(self.device)
            qt = task['query_targets'].to(self.device)

            # Adapt with no graph needed (eval only)
            fast_params = {k: v.clone().detach().requires_grad_(True)
                          for k, v in dict(self.model.named_parameters()).items()}
            with torch.backends.cudnn.flags(enabled=False):
                for _ in range(self.inner_steps):
                    logits = functional_call(self.model, fast_params, (ss,))
                    loss = self.compute_rank_loss(logits, st)
                    grads = torch.autograd.grad(loss, fast_params.values())
                    fast_params = {k: (v - self.inner_lr * g).detach().requires_grad_(True)
                                  for (k, v), g in zip(fast_params.items(), grads)}

            with torch.no_grad():
                q_logits = functional_call(self.model, fast_params, (qs,))
                q_loss = self.compute_rank_loss(q_logits, qt)
                acc = (q_logits.argmax(-1) == qt).float().mean().item()
                corr = self.compute_correlation(q_logits, qt)

            all_losses.append(q_loss.item())
            all_accs.append(acc)
            all_corrs.append(corr)

        return {
            'mean_loss': np.mean(all_losses),
            'mean_accuracy': np.mean(all_accs),
            'mean_correlation': np.mean(all_corrs),
        }


def main():
    parser = argparse.ArgumentParser(description='Numin2 MAML Training')
    parser.add_argument('--data_path', type=str, default='numin_sample.parquet')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--support_days', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints_numin')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = NuminLocalDataset(args.data_path, args.window_size, args.support_days, device)

    # Chronological split (no shuffling for time series)
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, n))

    print(f"Tasks: {n} total, {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    model = NuminRankModel(num_stocks=50, hidden_dim=args.hidden_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    maml = NuminMAML(model, args.inner_lr, args.outer_lr, args.inner_steps, device)
    os.makedirs(args.save_dir, exist_ok=True)

    best_val_corr = -float('inf')
    for epoch in range(args.epochs):
        epoch_losses, epoch_accs = [], []
        order = train_idx.copy()
        random.shuffle(order)

        pbar = tqdm(order, desc=f"Epoch {epoch+1}/{args.epochs}")
        for idx in pbar:
            task = dataset[idx]
            metrics = maml.train_step(task)
            epoch_losses.append(metrics['loss'])
            epoch_accs.append(metrics['accuracy'])
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.4f}")

        print(f"Epoch {epoch+1}: Loss={np.mean(epoch_losses):.4f}, Acc={np.mean(epoch_accs):.4f}")

        if (epoch + 1) % args.eval_interval == 0:
            val_results = maml.evaluate(dataset, val_idx)
            print(f"Validation - Loss: {val_results['mean_loss']:.4f}, "
                  f"Acc: {val_results['mean_accuracy']:.4f}, "
                  f"Corr: {val_results['mean_correlation']:.4f}")

            if val_results['mean_correlation'] > best_val_corr:
                best_val_corr = val_results['mean_correlation']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
                print(f"  Saved best model (val corr: {best_val_corr:.4f})")

    # Reload best model for test
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print("Loaded best model for testing")

    test_results = maml.evaluate(dataset, test_idx)
    print(f"Test - Loss: {test_results['mean_loss']:.4f}, "
          f"Acc: {test_results['mean_accuracy']:.4f}, "
          f"Corr: {test_results['mean_correlation']:.4f}")

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_corr': best_val_corr,
            'test_loss': test_results['mean_loss'],
            'test_accuracy': test_results['mean_accuracy'],
            'test_correlation': test_results['mean_correlation'],
            'args': vars(args)
        }, f, indent=2)

    print(f"\nDone! Best val corr: {best_val_corr:.4f}, Test corr: {test_results['mean_correlation']:.4f}")

if __name__ == '__main__':
    main()
