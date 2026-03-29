"""
1D-ARC with Prototypical Networks

Prototypical Networks learn a metric space where classification
can be performed by computing distances to prototype representations.
Uses per-position features and distance-based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import glob
import numpy as np
from tqdm import tqdm
import random
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ARC1DDataset:
    def __init__(self, data_dir, max_seq_len=100):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tasks = []

        task_dirs = glob.glob(os.path.join(data_dir, "1d_*"))
        for task_dir in task_dirs:
            if os.path.isdir(task_dir):
                task_files = glob.glob(os.path.join(task_dir, "*.json"))
                self.tasks.extend(task_files)

        print(f"Loaded {len(self.tasks)} tasks")

    def __len__(self):
        return len(self.tasks)

    def pad_sequence(self, seq, max_len):
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [10] * (max_len - len(seq))  # Fix A: padding_value=10

    def __getitem__(self, idx):
        with open(self.tasks[idx], 'r') as f:
            task_data = json.load(f)

        support_inputs, support_outputs, support_masks = [], [], []

        for example in task_data['train']:
            inp = example['input'][0] if isinstance(example['input'][0], list) else example['input']
            out = example['output'][0] if isinstance(example['output'][0], list) else example['output']
            if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list): inp = inp[0]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list): out = out[0]

            out_len = len(out)
            support_inputs.append(self.pad_sequence(inp, self.max_seq_len))
            support_outputs.append(self.pad_sequence(out, self.max_seq_len))
            support_masks.append([1]*min(out_len, self.max_seq_len) + [0]*max(0, self.max_seq_len-out_len))

        test_example = task_data['test'][0]
        test_inp = test_example['input'][0] if isinstance(test_example['input'][0], list) else test_example['input']
        test_out = test_example['output'][0] if isinstance(test_example['output'][0], list) else test_example['output']
        if isinstance(test_inp, list) and len(test_inp) > 0 and isinstance(test_inp[0], list): test_inp = test_inp[0]
        if isinstance(test_out, list) and len(test_out) > 0 and isinstance(test_out[0], list): test_out = test_out[0]

        test_out_len = len(test_out)

        return {
            'support_inputs': torch.tensor(support_inputs, dtype=torch.long),
            'support_outputs': torch.tensor(support_outputs, dtype=torch.long),
            'support_masks': torch.tensor(support_masks, dtype=torch.float),
            'query_input': torch.tensor(self.pad_sequence(test_inp, self.max_seq_len), dtype=torch.long),
            'query_output': torch.tensor(self.pad_sequence(test_out, self.max_seq_len), dtype=torch.long),
            'query_mask': torch.tensor([1]*min(test_out_len, self.max_seq_len) + [0]*max(0, self.max_seq_len-test_out_len), dtype=torch.float),
        }


class ProtoNetEncoder(nn.Module):
    """Encoder that creates per-position embeddings."""

    def __init__(self, vocab_size=11, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=10)  # Fix A: vocab_size=11, padding_idx=10

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        x = self.output_proj(x)  # (batch, seq_len, hidden_dim)
        return x


class ProtoNet(nn.Module):
    """Prototypical Network for 1D-ARC using distance-based classification."""

    def __init__(self, vocab_size=11, embed_dim=128, hidden_dim=256, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ProtoNetEncoder(vocab_size, embed_dim, hidden_dim)

    def embedding_per_position(self, x):
        """Get per-position embeddings."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x).squeeze(0)  # (L, hidden) if single, (batch, L, hidden) if batch

    def compute_prototypes(self, support_inputs, support_outputs, support_masks):
        """Compute per-class prototypes from support examples."""
        # Encode support inputs to get per-position features
        per_pos_features = self.encoder(support_inputs)  # (K, L, hidden)

        # Build prototype per class (0-9)
        prototypes = torch.zeros(self.num_classes, per_pos_features.size(-1), device=per_pos_features.device)
        counts = torch.zeros(self.num_classes, device=per_pos_features.device)

        for i in range(support_inputs.size(0)):
            for pos in range(support_outputs.size(1)):
                if support_masks[i, pos] > 0:  # only non-padding
                    cls = support_outputs[i, pos].item()
                    if cls < self.num_classes:  # valid class
                        prototypes[cls] += per_pos_features[i, pos]
                        counts[cls] += 1

        # Average
        valid = counts > 0
        prototypes[valid] /= counts[valid].unsqueeze(1)
        return prototypes

    def forward(self, support_inputs, support_outputs, query_input, support_masks=None, query_mask=None):
        prototypes = self.compute_prototypes(support_inputs, support_outputs, support_masks)  # (num_classes, hidden)

        if query_input.dim() == 1:
            query_input = query_input.unsqueeze(0)
        query_features = self.encoder(query_input).squeeze(0)  # (L, hidden)

        # Distance-based classification: for each position, compute distance to each prototype
        # logits = negative squared Euclidean distance
        logits = -torch.cdist(query_features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0) ** 2
        # logits shape: (L, num_classes)
        return logits


def train_epoch(model, dataset, optimizer, device, batch_indices):
    model.train()
    total_loss = 0

    for idx in tqdm(batch_indices, desc="Training"):
        batch = dataset[idx]

        support_inputs = batch['support_inputs'].to(device)
        support_outputs = batch['support_outputs'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_input = batch['query_input'].to(device)
        query_output = batch['query_output'].to(device)
        query_mask = batch['query_mask'].to(device)

        optimizer.zero_grad()
        logits = model(support_inputs, support_outputs, query_input,
                       support_masks=support_masks, query_mask=query_mask)

        loss = F.cross_entropy(logits, query_output, reduction='none', ignore_index=10)
        loss = (loss * query_mask).sum() / query_mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Fix E: gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batch_indices)


def evaluate(model, dataset, device, indices):
    model.eval()
    correct, total = 0, 0
    task_accs = []

    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
            batch = dataset[idx]

            support_inputs = batch['support_inputs'].to(device)
            support_outputs = batch['support_outputs'].to(device)
            support_masks = batch['support_masks'].to(device)
            query_input = batch['query_input'].to(device)
            query_output = batch['query_output'].to(device)
            query_mask = batch['query_mask'].to(device)

            logits = model(support_inputs, support_outputs, query_input,
                           support_masks=support_masks, query_mask=query_mask)
            preds = logits.argmax(dim=-1)

            mask = query_mask.bool()
            task_correct = (preds[mask] == query_output[mask]).sum().item()
            task_total = mask.sum().item()

            correct += task_correct
            total += task_total
            task_accs.append(task_correct / task_total if task_total > 0 else 0)

    return {'overall_acc': correct/total, 'mean_task_acc': np.mean(task_accs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='1D-ARC/dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_protonet')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (-1 for CPU)')  # Fix D
    args = parser.parse_args()

    set_seed(args.seed)

    # Fix D: GPU selection
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = ARC1DDataset(args.data_dir)

    # Split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = ProtoNet(vocab_size=11, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=10).to(device)  # Fix A
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        loss = train_epoch(model, dataset, optimizer, device, train_idx)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        if (epoch + 1) % 10 == 0:
            val_results = evaluate(model, dataset, device, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_task_acc'] > best_val_acc:
                best_val_acc = val_results['mean_task_acc']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
                print(f"Saved best model: {best_val_acc:.4f}")

    # Fix C: Reload best model before test
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for testing.")

    # Final test
    test_results = evaluate(model, dataset, device, test_idx)
    print(f"\nTest Results: {test_results}")

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'test_overall_acc': test_results['overall_acc'],
            'test_mean_task_acc': test_results['mean_task_acc'],
            'args': vars(args)
        }, f, indent=2)

    print("Training complete!")

if __name__ == '__main__':
    main()
