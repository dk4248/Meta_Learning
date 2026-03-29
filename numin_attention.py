"""
Numin2 with Stock Attention MAML

Key idea: Stocks influence each other. Use attention to model
stock-to-stock relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import random
import argparse
import json
import os

# Disable efficient SDPA for double backward compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

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

        task_list = []
        for (year, month), group in df.groupby(['year', 'month']):
            group = group.sort_index()
            returns = group.drop(['year', 'month'], axis=1).values

            if len(returns) < window_size + support_days + 1:
                continue

            samples, targets = [], []
            for i in range(window_size, len(returns)):
                window = returns[i - window_size:i].copy()
                ranks = np.argsort(np.argsort(-returns[i]))
                samples.append(window)
                targets.append(ranks)

            if len(samples) >= support_days + 1:
                task_list.append((year, month, {
                    'samples': np.array(samples, dtype=np.float32),
                    'targets': np.array(targets, dtype=np.int64)
                }))

        # Sort tasks chronologically
        task_list.sort(key=lambda x: (x[0], x[1]))
        self.tasks = [t[2] for t in task_list]

        print(f"Created {len(self.tasks)} tasks")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        samples = task['samples'].copy()
        targets = task['targets'].copy()

        support_idx = list(range(min(self.support_days, len(samples)-1)))
        query_idx = list(range(len(support_idx), len(samples)))

        # Normalize using support statistics only
        support_samples = samples[support_idx].copy()
        mean = support_samples.mean()
        std = support_samples.std() + 1e-8
        samples = (samples - mean) / std

        return {
            'support_samples': torch.tensor(samples[support_idx]),
            'support_targets': torch.tensor(targets[support_idx]),
            'query_samples': torch.tensor(samples[query_idx]),
            'query_targets': torch.tensor(targets[query_idx]),
        }


class StockAttentionModel(nn.Module):
    """
    Model with attention across stocks to capture inter-stock relationships.
    """

    def __init__(self, num_stocks=50, window_size=50, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_stocks = num_stocks

        # Per-stock temporal encoder
        self.temporal_encoder = nn.LSTM(1, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)

        # Stock embedding
        self.stock_embedding = nn.Embedding(num_stocks, hidden_dim)

        # Cross-stock attention
        self.stock_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks)
        )

    def forward(self, x):
        # x: (batch, window, num_stocks)
        batch_size = x.size(0)
        window_size = x.size(1)

        # Flatten LSTM parameters for efficiency
        self.temporal_encoder.flatten_parameters()

        # Process each stock's time series separately
        # Reshape to (batch * num_stocks, window, 1)
        x_reshaped = x.permute(0, 2, 1).reshape(batch_size * self.num_stocks, window_size, 1)

        # Encode temporal patterns per stock
        temporal_out, _ = self.temporal_encoder(x_reshaped)  # (batch*stocks, window, hidden)
        temporal_out = temporal_out.mean(dim=1)  # (batch*stocks, hidden)
        temporal_out = temporal_out.view(batch_size, self.num_stocks, -1)  # (batch, stocks, hidden)

        # Add stock embeddings
        stock_ids = torch.arange(self.num_stocks, device=x.device)
        stock_emb = self.stock_embedding(stock_ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = temporal_out + stock_emb

        # Cross-stock attention
        attn_out, _ = self.stock_attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # Decode to ranks
        logits = self.decoder(x)  # (batch, stocks, num_stocks)

        return logits


class AttentionMAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, device='cuda'):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def compute_loss(self, logits, targets):
        B, S, C = logits.shape
        return F.cross_entropy(logits.reshape(B * S, C), targets.reshape(B * S))

    def train_step(self, task):
        self.model.train()
        # Disable cuDNN for LSTM double backward compatibility
        torch.backends.cudnn.enabled = False

        ss = task['support_samples'].to(self.device)
        st = task['support_targets'].to(self.device)
        qs = task['query_samples'].to(self.device)
        qt = task['query_targets'].to(self.device)

        fast_params = {k: v.clone() for k, v in dict(self.model.named_parameters()).items()}

        for _ in range(self.inner_steps):
            logits = functional_call(self.model, fast_params, (ss,))
            loss = self.compute_loss(logits, st)
            grads = torch.autograd.grad(loss, fast_params.values(), create_graph=True)
            fast_params = {k: v - self.inner_lr * g for (k, v), g in zip(fast_params.items(), grads)}

        query_logits = functional_call(self.model, fast_params, (qs,))
        query_loss = self.compute_loss(query_logits, qt)

        self.optimizer.zero_grad()
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        acc = (query_logits.argmax(dim=-1) == qt).float().mean().item()
        torch.backends.cudnn.enabled = True
        return {'loss': query_loss.item(), 'accuracy': acc}

    def evaluate(self, dataset, indices):
        self.model.train()  # Keep train mode for LSTM backward compatibility
        all_corrs = []

        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]

            support_samples = task['support_samples'].to(self.device)
            support_targets = task['support_targets'].to(self.device)
            query_samples = task['query_samples'].to(self.device)
            query_targets = task['query_targets'].to(self.device)

            # Use functional_call for evaluation inner loop (no graph needed)
            fast_params = {k: v.clone().detach().requires_grad_(True) for k, v in dict(self.model.named_parameters()).items()}

            for _ in range(self.inner_steps):
                logits = functional_call(self.model, fast_params, (support_samples,))
                loss = self.compute_loss(logits, support_targets)
                grads = torch.autograd.grad(loss, fast_params.values())
                fast_params = {k: (v - self.inner_lr * g).detach().requires_grad_(True)
                               for (k, v), g in zip(fast_params.items(), grads)}

            with torch.no_grad():
                query_logits = functional_call(self.model, fast_params, (query_samples,))
                preds = query_logits.argmax(dim=-1).cpu().numpy()
                tgts = query_targets.cpu().numpy()

                for i in range(len(preds)):
                    corr, _ = spearmanr(preds[i], tgts[i])
                    if not np.isnan(corr):
                        all_corrs.append(corr)

        return {'mean_correlation': np.mean(all_corrs) if all_corrs else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='numin_sample.parquet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.0005)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_numin_attention')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use')
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = NuminDataset(args.data_path, window_size=args.window_size)

    # Temporal (chronological) split -- tasks are already sorted by time
    num_tasks = len(dataset)
    train_size = int(0.7 * num_tasks)
    val_size = int(0.15 * num_tasks)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, num_tasks))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = StockAttentionModel(num_stocks=50, window_size=args.window_size, hidden_dim=args.hidden_dim)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    maml = AttentionMAML(model, args.inner_lr, args.outer_lr, args.inner_steps, device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_corr = -1
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        losses, accs = [], []

        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}"):
            metrics = maml.train_step(dataset[idx])
            losses.append(metrics['loss'])
            accs.append(metrics['accuracy'])

        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f}, Acc={np.mean(accs):.4f}")

        if (epoch + 1) % 5 == 0:
            val_results = maml.evaluate(dataset, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_correlation'] > best_val_corr:
                best_val_corr = val_results['mean_correlation']
                torch.save(model.state_dict(), best_model_path)

    # Reload best model before test evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for test evaluation.")

    test_results = maml.evaluate(dataset, test_idx)
    print(f"\nTest: {test_results}")

    results = {
        'best_val_corr': best_val_corr,
        'test_correlation': test_results['mean_correlation'],
        'args': vars(args)
    }
    results_path = os.path.join(args.save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest Val Corr: {best_val_corr:.4f}")
    print(f"Test Corr: {test_results['mean_correlation']:.4f}")

if __name__ == '__main__':
    main()
