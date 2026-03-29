"""
1D-ARC Reptile Meta-Learning

Reptile: Simpler meta-learning that often works better.
1. Sample a task
2. Train on task for k steps
3. Move model weights towards trained weights: theta = theta + epsilon(theta' - theta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
import numpy as np
from tqdm import tqdm
import random
import argparse

PAD_VALUE = 10


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ARC1DDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=100):
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
        return seq + [PAD_VALUE] * (max_len - len(seq))

    def __getitem__(self, idx):
        with open(self.tasks[idx], 'r') as f:
            task_data = json.load(f)

        support_inputs, support_outputs, support_masks = [], [], []

        for example in task_data['train']:
            inp = example['input'][0] if isinstance(example['input'][0], list) else example['input']
            out = example['output'][0] if isinstance(example['output'][0], list) else example['output']

            if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list):
                inp = inp[0]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                out = out[0]

            out_len = len(out)
            support_inputs.append(self.pad_sequence(inp, self.max_seq_len))
            support_outputs.append(self.pad_sequence(out, self.max_seq_len))
            support_masks.append([1] * min(out_len, self.max_seq_len) + [0] * max(0, self.max_seq_len - out_len))

        test_example = task_data['test'][0]
        test_inp = test_example['input'][0] if isinstance(test_example['input'][0], list) else test_example['input']
        test_out = test_example['output'][0] if isinstance(test_example['output'][0], list) else test_example['output']

        if isinstance(test_inp, list) and len(test_inp) > 0 and isinstance(test_inp[0], list):
            test_inp = test_inp[0]
        if isinstance(test_out, list) and len(test_out) > 0 and isinstance(test_out[0], list):
            test_out = test_out[0]

        test_out_len = len(test_out)
        query_input = self.pad_sequence(test_inp, self.max_seq_len)
        query_output = self.pad_sequence(test_out, self.max_seq_len)
        query_mask = [1] * min(test_out_len, self.max_seq_len) + [0] * max(0, self.max_seq_len - test_out_len)

        return {
            'support_inputs': torch.tensor(support_inputs, dtype=torch.long),
            'support_outputs': torch.tensor(support_outputs, dtype=torch.long),
            'support_masks': torch.tensor(support_masks, dtype=torch.float),
            'query_input': torch.tensor(query_input, dtype=torch.long),
            'query_output': torch.tensor(query_output, dtype=torch.long),
            'query_mask': torch.tensor(query_mask, dtype=torch.float),
        }


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size=11, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_VALUE)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.norm3(x)
        return x


class ARC1DModel(nn.Module):
    def __init__(self, vocab_size=11, embed_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.encoder = SequenceEncoder(vocab_size, embed_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        self.hidden_dim = hidden_dim

    def encode_examples(self, support_inputs, support_outputs):
        inp_enc = self.encoder(support_inputs)
        out_enc = self.encoder(support_outputs)
        combined = torch.cat([inp_enc, out_enc], dim=1)
        example_enc = combined.mean(dim=0)
        return example_enc

    def forward(self, support_inputs, support_outputs, query_input):
        if query_input.dim() == 1:
            query_input = query_input.unsqueeze(0)

        example_enc = self.encode_examples(support_inputs, support_outputs)
        example_enc = example_enc.unsqueeze(0)
        query_enc = self.encoder(query_input)
        attended, _ = self.cross_attention(query_enc, example_enc, example_enc)
        combined = torch.cat([query_enc, attended], dim=-1)
        logits = self.decoder(combined)
        return logits.squeeze(0)


class Reptile:
    def __init__(self, model, outer_lr=0.1, inner_lr=0.01, inner_steps=10, device='cuda'):
        self.model = model.to(device)
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device

    def train_step(self, task):
        self.model.train()

        support_inputs = task['support_inputs'].to(self.device)
        support_outputs = task['support_outputs'].to(self.device)
        support_masks = task['support_masks'].to(self.device)
        K = support_inputs.size(0)

        # Save original weights
        old_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # Inner loop: train on support set
        inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            inner_loss = 0
            for i in range(K):
                # Leave-one-out context
                ctx_idx = [k for k in range(K) if k != i]
                if len(ctx_idx) > 0:
                    ctx_in = support_inputs[ctx_idx]
                    ctx_out = support_outputs[ctx_idx]
                else:
                    ctx_in = support_inputs[i:i+1]
                    ctx_out = torch.zeros_like(support_outputs[i:i+1])
                logits = self.model(ctx_in, ctx_out, support_inputs[i])
                ce_loss = F.cross_entropy(logits, support_outputs[i], reduction='none', ignore_index=PAD_VALUE)
                inner_loss += (ce_loss * support_masks[i]).sum() / support_masks[i].sum()

            inner_loss = inner_loss / K
            inner_opt.zero_grad()
            inner_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            inner_opt.step()

        # Evaluate on query BEFORE Reptile interpolation (model has fully-adapted weights)
        query_input = task['query_input'].to(self.device)
        query_output = task['query_output'].to(self.device)
        query_mask = task['query_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            query_logits = self.model(support_inputs, support_outputs, query_input)
            predictions = query_logits.argmax(dim=-1)
            mask = query_mask.bool()
            acc = (predictions[mask] == query_output[mask]).float().mean().item()

        # Reptile update: move towards trained weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = old_weights[name] + self.outer_lr * (param.data - old_weights[name])

        return {'accuracy': acc}

    def evaluate(self, dataset, indices):
        all_accs = []

        for idx in tqdm(indices, desc="Evaluating"):
            task = dataset[idx]

            support_inputs = task['support_inputs'].to(self.device)
            support_outputs = task['support_outputs'].to(self.device)
            support_masks = task['support_masks'].to(self.device)
            query_input = task['query_input'].to(self.device)
            query_output = task['query_output'].to(self.device)
            query_mask = task['query_mask'].to(self.device)
            K = support_inputs.size(0)

            # Save weights
            old_weights = {name: param.clone() for name, param in self.model.named_parameters()}

            # Adapt
            inner_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            self.model.train()

            for _ in range(self.inner_steps):
                inner_loss = 0
                for i in range(K):
                    # Leave-one-out context
                    ctx_idx = [k for k in range(K) if k != i]
                    if len(ctx_idx) > 0:
                        ctx_in = support_inputs[ctx_idx]
                        ctx_out = support_outputs[ctx_idx]
                    else:
                        ctx_in = support_inputs[i:i+1]
                        ctx_out = torch.zeros_like(support_outputs[i:i+1])
                    logits = self.model(ctx_in, ctx_out, support_inputs[i])
                    ce_loss = F.cross_entropy(logits, support_outputs[i], reduction='none', ignore_index=PAD_VALUE)
                    inner_loss += (ce_loss * support_masks[i]).sum() / support_masks[i].sum()

                inner_loss = inner_loss / K
                inner_opt.zero_grad()
                inner_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                inner_opt.step()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                query_logits = self.model(support_inputs, support_outputs, query_input)
                predictions = query_logits.argmax(dim=-1)
                mask = query_mask.bool()
                acc = (predictions[mask] == query_output[mask]).float().mean().item()
                all_accs.append(acc)

            # Restore weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = old_weights[name]

        return {'mean_accuracy': np.mean(all_accs) if all_accs else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='1D-ARC/dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--inner_lr', type=float, default=0.01)
    parser.add_argument('--outer_lr', type=float, default=0.1)
    parser.add_argument('--inner_steps', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints_arc_reptile')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
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

    model = ARC1DModel(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    reptile = Reptile(model, args.outer_lr, args.inner_lr, args.inner_steps, device)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        random.shuffle(train_idx)
        accs = []

        # Decay outer_lr
        progress = epoch / num_epochs
        reptile.outer_lr = args.outer_lr * (1.0 - progress)

        for idx in tqdm(train_idx, desc=f"Epoch {epoch+1}"):
            metrics = reptile.train_step(dataset[idx])
            accs.append(metrics['accuracy'])

        print(f"Epoch {epoch+1}: Acc={np.mean(accs):.4f} outer_lr={reptile.outer_lr:.5f}")

        if (epoch + 1) % 5 == 0:
            val_results = reptile.evaluate(dataset, val_idx)
            print(f"Validation: {val_results}")

            if val_results['mean_accuracy'] > best_val_acc:
                best_val_acc = val_results['mean_accuracy']
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))

    # Test - reload best model
    best_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model for test evaluation.")

    test_results = reptile.evaluate(dataset, test_idx)
    print(f"\nTest: {test_results}")

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'test_accuracy': test_results['mean_accuracy'],
            'args': vars(args)
        }, f, indent=2)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_results['mean_accuracy']:.4f}")


if __name__ == '__main__':
    main()
