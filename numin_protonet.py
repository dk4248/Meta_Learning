"""
Numin2 with Prototypical Networks

Instead of adapting weights, learn a metric space where similar
market conditions cluster together.
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


class ProtoNetEncoder(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.proj(out)


class NuminProtoNet(nn.Module):
    def __init__(self, num_stocks=50, hidden_dim=256):
        super().__init__()
        self.encoder = ProtoNetEncoder(num_stocks, hidden_dim)

        # Learned rank embeddings
        self.rank_embedding = nn.Embedding(num_stocks, hidden_dim)

        # Projects support encoding + rank embedding into a prototype contribution
        self.proto_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_stocks)
        )

        self.num_stocks = num_stocks
        self.hidden_dim = hidden_dim

    def forward(self, support_samples, support_targets, query_samples):
        ns = self.num_stocks

        # Encode support samples: (K, hidden)
        support_enc = self.encoder(support_samples)
        K = support_enc.size(0)

        # Build per-rank prototypes using the actual stock-rank assignments.
        # For each support sample i and each stock s, the stock is assigned
        # rank r = support_targets[i, s].  We combine the support encoding
        # with the rank embedding and accumulate into the prototype for rank r.
        rank_emb = self.rank_embedding.weight  # (ns, hidden)

        # Gather rank indices per support sample: (K, ns) -> flatten
        flat_ranks = support_targets.view(-1)  # (K*ns,)

        # Expand support encodings per stock: each of K encodings repeated ns times
        support_exp = support_enc.unsqueeze(1).expand(K, ns, -1).reshape(K * ns, -1)  # (K*ns, hidden)

        # Gather rank embeddings for the assigned ranks
        rank_emb_gathered = rank_emb[flat_ranks]  # (K*ns, hidden)

        # Project concatenation of support encoding and rank embedding
        proto_input = torch.cat([support_exp, rank_emb_gathered], dim=-1)  # (K*ns, hidden*2)
        proto_contrib = self.proto_proj(proto_input)  # (K*ns, hidden)

        # Scatter-add into prototypes per rank
        prototypes = torch.zeros(ns, self.hidden_dim, device=support_samples.device)
        counts = torch.zeros(ns, 1, device=support_samples.device)

        idx_exp = flat_ranks.unsqueeze(1).expand(-1, self.hidden_dim)  # (K*ns, hidden)
        prototypes.scatter_add_(0, idx_exp, proto_contrib)
        counts.scatter_add_(0, flat_ranks.unsqueeze(1), torch.ones(K * ns, 1, device=support_samples.device))

        counts = counts.clamp(min=1)
        prototypes = prototypes / counts  # (ns, hidden)

        # Encode queries: (B, hidden)
        query_enc = self.encoder(query_samples)

        # For each query, concatenate with each prototype and decode
        # query_enc: (B, hidden), prototypes: (ns, hidden)
        B = query_enc.size(0)
        q_exp = query_enc.unsqueeze(1).expand(B, ns, -1)  # (B, ns, hidden)
        p_exp = prototypes.unsqueeze(0).expand(B, ns, -1)  # (B, ns, hidden)
        combined = torch.cat([q_exp, p_exp], dim=-1)  # (B, ns, hidden*2)
        logits = self.decoder(combined)  # (B, ns, ns)

        return logits


def train_epoch(model, dataset, optimizer, device, indices):
    model.train()
    total_loss = 0
    total_acc = 0

    for idx in tqdm(indices, desc="Training"):
        task = dataset[idx]

        support_samples = task['support_samples'].to(device)
        support_targets = task['support_targets'].to(device)
        query_samples = task['query_samples'].to(device)
        query_targets = task['query_targets'].to(device)

        optimizer.zero_grad()
        logits = model(support_samples, support_targets, query_samples)

        # Vectorized cross entropy loss
        loss = F.cross_entropy(logits.view(-1, model.num_stocks), query_targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        acc = (logits.argmax(dim=-1) == query_targets).float().mean().item()
        total_acc += acc

    return total_loss / len(indices), total_acc / len(indices)


def evaluate(model, dataset, device, indices):
    model.eval()
    all_corrs = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]

            support_samples = task['support_samples'].to(device)
            support_targets = task['support_targets'].to(device)
            query_samples = task['query_samples'].to(device)
            query_targets = task['query_targets'].to(device)

            logits = model(support_samples, support_targets, query_samples)
            preds = logits.argmax(dim=-1).cpu().numpy()
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_numin_protonet')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = NuminDataset(args.data_path, window_size=args.window_size)

    # Temporal split (tasks already sorted chronologically)
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, n))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = NuminProtoNet(num_stocks=50, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_corr = -1

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        loss, acc = train_epoch(model, dataset, optimizer, device, train_idx)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

        if (epoch + 1) % 5 == 0:
            val_results = evaluate(model, dataset, device, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_correlation'] > best_val_corr:
                best_val_corr = val_results['mean_correlation']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

    # Reload best model before test
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_results = evaluate(model, dataset, device, test_idx)
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
