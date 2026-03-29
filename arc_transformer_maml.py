"""
1D-ARC with Transformer-based MAML

Uses Transformer architecture for better sequence modeling.
Fixed: proper MAML with functional_call, sinusoidal positional encoding,
leave-one-out inner loop, and common fixes A-F.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
import json
import os
import glob
import numpy as np
import math
from copy import deepcopy
from tqdm import tqdm
import random
import argparse

# Disable efficient SDPA for double backward compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

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
                self.tasks.extend(glob.glob(os.path.join(task_dir, "*.json")))
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


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that works for any sequence length."""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]  # (seq_len, embed_dim)


class TransformerARC(nn.Module):
    """Transformer-based model for 1D-ARC."""

    def __init__(self, vocab_size=11, embed_dim=128, hidden_dim=256, num_heads=4, num_layers=4, num_classes=10, max_seq_len=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=10)  # Fix A: vocab_size=11, padding_idx=10
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=5000)  # Fix: sinusoidal, handles any length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

        self.max_seq_len = max_seq_len

    def forward(self, support_inputs, support_outputs, query_input):
        # Concatenate all sequences: [support_in_1, support_out_1, ..., query]
        batch_size = support_inputs.size(0)
        seq_len = support_inputs.size(1)

        if query_input.dim() == 1:
            query_input = query_input.unsqueeze(0)

        # Flatten support pairs
        sequences = []
        for i in range(batch_size):
            sequences.append(support_inputs[i])
            sequences.append(support_outputs[i])
        sequences.append(query_input.squeeze(0))

        # Stack all sequences
        all_seq = torch.stack(sequences, dim=0)  # (2*n_support + 1, seq_len)

        # Embed tokens
        x = self.embedding(all_seq)  # (num_seqs, seq_len, embed_dim)

        # Add sinusoidal positional encoding across the full flattened sequence
        total_len = all_seq.size(0) * seq_len
        pos_enc = self.pos_encoding(total_len)  # (total_len, embed_dim)

        # Flatten to single sequence for transformer
        x = x.view(1, total_len, -1)  # (1, total_len, embed_dim)
        x = x + pos_enc.unsqueeze(0)  # add positional encoding

        # Transform
        x = self.transformer(x)

        # Extract query output (last seq_len positions)
        query_out = x[0, -seq_len:, :]  # (seq_len, embed_dim)

        # Decode
        logits = self.decoder(query_out)  # (seq_len, num_classes)

        return logits


class TransformerMAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, device='cuda'):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def _forward_with_params(self, params, support_inputs, support_outputs, query_input):
        """Forward pass using functional_call for proper MAML gradient flow."""
        return functional_call(self.model, params, (support_inputs, support_outputs, query_input))

    def train_step(self, batch):
        self.model.train()

        support_inputs = batch['support_inputs'].to(self.device)
        support_outputs = batch['support_outputs'].to(self.device)
        support_masks = batch['support_masks'].to(self.device)
        query_input = batch['query_input'].to(self.device)
        query_output = batch['query_output'].to(self.device)
        query_mask = batch['query_mask'].to(self.device)

        # Create fast parameters (copy of model params with grad tracking)
        fast_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Inner loop with leave-one-out
        n_support = support_inputs.size(0)
        for _ in range(self.inner_steps):
            inner_loss = 0

            for i in range(n_support):
                # Fix: Leave-one-out - train on all except i, predict i
                loo_mask = torch.ones(n_support, dtype=torch.bool, device=self.device)
                loo_mask[i] = False
                loo_inputs = support_inputs[loo_mask]
                loo_outputs = support_outputs[loo_mask]

                # Predict held-out example i
                logits = self._forward_with_params(fast_params, loo_inputs, loo_outputs, support_inputs[i])
                loss = F.cross_entropy(logits, support_outputs[i], reduction='none', ignore_index=10)
                inner_loss += (loss * support_masks[i]).sum() / support_masks[i].sum()

            inner_loss = inner_loss / n_support

            # Manual SGD update on fast_params
            grads = torch.autograd.grad(inner_loss, fast_params.values(), create_graph=True)
            fast_params = {name: param - self.inner_lr * grad
                           for (name, param), grad in zip(fast_params.items(), grads)}

        # Query loss using adapted parameters
        query_logits = self._forward_with_params(fast_params, support_inputs, support_outputs, query_input)
        query_loss = F.cross_entropy(query_logits, query_output, reduction='none', ignore_index=10)
        query_loss = (query_loss * query_mask).sum() / query_mask.sum()

        self.optimizer.zero_grad()
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Fix E: gradient clipping
        self.optimizer.step()

        return query_loss.item()

    def evaluate(self, dataset, indices):
        self.model.eval()
        correct, total = 0, 0
        task_accs = []

        for idx in tqdm(indices, desc="Evaluating"):
            batch = dataset[idx]

            support_inputs = batch['support_inputs'].to(self.device)
            support_outputs = batch['support_outputs'].to(self.device)
            support_masks = batch['support_masks'].to(self.device)
            query_input = batch['query_input'].to(self.device)
            query_output = batch['query_output'].to(self.device)
            query_mask = batch['query_mask'].to(self.device)

            # Adapt with leave-one-out (no grad needed at eval)
            fast_params = {name: param.clone().detach().requires_grad_(True)
                           for name, param in self.model.named_parameters()}

            n_support = support_inputs.size(0)
            for _ in range(self.inner_steps):
                inner_loss = 0
                for i in range(n_support):
                    # Leave-one-out
                    loo_mask = torch.ones(n_support, dtype=torch.bool, device=self.device)
                    loo_mask[i] = False
                    loo_inputs = support_inputs[loo_mask]
                    loo_outputs = support_outputs[loo_mask]

                    logits = self._forward_with_params(fast_params, loo_inputs, loo_outputs, support_inputs[i])
                    loss = F.cross_entropy(logits, support_outputs[i], reduction='none', ignore_index=10)
                    inner_loss += (loss * support_masks[i]).sum() / support_masks[i].sum()
                inner_loss /= n_support

                grads = torch.autograd.grad(inner_loss, fast_params.values())
                fast_params = {name: (param - self.inner_lr * grad).detach().requires_grad_(True)
                               for (name, param), grad in zip(fast_params.items(), grads)}

            with torch.no_grad():
                logits = self._forward_with_params(fast_params, support_inputs, support_outputs, query_input)
                preds = logits.argmax(dim=-1)
                mask = query_mask.bool()
                tc = (preds[mask] == query_output[mask]).sum().item()
                tt = mask.sum().item()
                correct += tc
                total += tt
                task_accs.append(tc/tt if tt > 0 else 0)

        return {'overall_acc': correct/total if total > 0 else 0, 'mean_task_acc': np.mean(task_accs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='1D-ARC/dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.0005)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_transformer_maml')
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

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = TransformerARC(
        vocab_size=11, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,  # Fix A: vocab_size=11
        num_layers=args.num_layers
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    maml = TransformerMAML(model, inner_lr=args.inner_lr, outer_lr=args.outer_lr,
                           inner_steps=args.inner_steps, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0

    for epoch in range(args.epochs):
        random.shuffle(train_idx)
        losses = []

        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}"):
            batch = dataset[idx]
            loss = maml.train_step(batch)
            losses.append(loss)

        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")

        if (epoch + 1) % 10 == 0:
            val_results = maml.evaluate(dataset, val_idx[:50])
            print(f"Validation: {val_results}")

            if val_results['mean_task_acc'] > best_val_acc:
                best_val_acc = val_results['mean_task_acc']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

    # Fix C: Reload best model before test
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for testing.")

    test_results = maml.evaluate(dataset, test_idx)
    print(f"\nTest Results: {test_results}")

    # Fix F: already using with open()
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'test_overall_acc': test_results['overall_acc'],
            'test_mean_task_acc': test_results['mean_task_acc'],
            'args': vars(args)
        }, f, indent=2)

if __name__ == '__main__':
    main()
