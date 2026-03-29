"""
1D-ARC Meta-Learning with MAML + Hebbian TTA

This implements:
1. MAML (Model-Agnostic Meta-Learning) for few-shot learning on 1D-ARC tasks
2. Hebbian Test-Time Adaptation for improved generalization

Each 1D-ARC task has 3 training examples and 1 test example.
The model learns to adapt quickly to new tasks using the training examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.func import functional_call
import json
import os
import glob
import numpy as np
from tqdm import tqdm
import random
import argparse

# Disable efficient SDPA for double backward compatibility
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Padding token index - ARC uses values 0-9, so 10 is safe for padding
PAD_IDX = 10
VOCAB_SIZE = 11  # 10 colors (0-9) + 1 padding token (10)
NUM_CLASSES = 10  # We predict classes 0-9 only

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ARC1DDataset(Dataset):
    """Dataset for 1D-ARC tasks."""

    def __init__(self, data_dir, task_types=None, max_seq_len=100):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tasks = []

        # Load all tasks
        if task_types is None:
            task_dirs = glob.glob(os.path.join(data_dir, "1d_*"))
        else:
            task_dirs = [os.path.join(data_dir, t) for t in task_types]

        for task_dir in task_dirs:
            if os.path.isdir(task_dir):
                task_files = glob.glob(os.path.join(task_dir, "*.json"))
                for task_file in task_files:
                    self.tasks.append(task_file)

        print(f"Loaded {len(self.tasks)} tasks from {len(task_dirs)} task types")

    def __len__(self):
        return len(self.tasks)

    def pad_sequence(self, seq, max_len):
        """Pad sequence to max_len with PAD_IDX (10)."""
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [PAD_IDX] * (max_len - len(seq))

    def __getitem__(self, idx):
        with open(self.tasks[idx], 'r') as f:
            task_data = json.load(f)

        # Process training examples (support set)
        support_inputs = []
        support_outputs = []
        support_masks = []

        for example in task_data['train']:
            inp = example['input'][0] if isinstance(example['input'][0], list) else example['input']
            out = example['output'][0] if isinstance(example['output'][0], list) else example['output']

            # Handle nested lists
            if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], list):
                inp = inp[0]
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                out = out[0]

            inp_len = len(inp)
            out_len = len(out)

            support_inputs.append(self.pad_sequence(inp, self.max_seq_len))
            support_outputs.append(self.pad_sequence(out, self.max_seq_len))
            support_masks.append([1] * min(out_len, self.max_seq_len) + [0] * max(0, self.max_seq_len - out_len))

        # Process test example (query set)
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
            'task_file': self.tasks[idx]
        }


class SequenceEncoder(nn.Module):
    """Encoder for 1D sequences using 1D convolutions."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

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
        x = self.norm3(x)  # (batch, seq_len, hidden_dim)

        return x


class ARC1DModel(nn.Module):
    """
    Model for 1D-ARC tasks.
    Uses example encoding + query encoding to predict output sequence.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()

        self.encoder = SequenceEncoder(vocab_size, embed_dim, hidden_dim)

        # Cross-attention to attend to examples
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Decoder
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
        """
        Encode support set examples.
        support_inputs: (num_examples, seq_len)
        support_outputs: (num_examples, seq_len)
        """
        # Encode inputs and outputs
        inp_enc = self.encoder(support_inputs)  # (num_examples, seq_len, hidden_dim)
        out_enc = self.encoder(support_outputs)  # (num_examples, seq_len, hidden_dim)

        # Combine input and output encodings
        combined = torch.cat([inp_enc, out_enc], dim=1)  # (num_examples, 2*seq_len, hidden_dim)

        # Pool across examples
        example_enc = combined.mean(dim=0)  # (2*seq_len, hidden_dim)

        return example_enc

    def forward(self, support_inputs, support_outputs, query_input):
        """
        Forward pass for a single task.
        support_inputs: (num_examples, seq_len)
        support_outputs: (num_examples, seq_len)
        query_input: (seq_len,) or (1, seq_len)
        """
        if query_input.dim() == 1:
            query_input = query_input.unsqueeze(0)

        # Encode examples
        example_enc = self.encode_examples(support_inputs, support_outputs)  # (2*seq_len, hidden_dim)
        example_enc = example_enc.unsqueeze(0)  # (1, 2*seq_len, hidden_dim)

        # Encode query
        query_enc = self.encoder(query_input)  # (1, seq_len, hidden_dim)

        # Cross-attention: query attends to examples
        attended, _ = self.cross_attention(query_enc, example_enc, example_enc)

        # Combine query encoding with attended features
        combined = torch.cat([query_enc, attended], dim=-1)  # (1, seq_len, 2*hidden_dim)

        # Decode to output
        logits = self.decoder(combined)  # (1, seq_len, num_classes)

        return logits.squeeze(0)  # (seq_len, num_classes)


class HebbianTTA(nn.Module):
    """
    Hebbian Test-Time Adaptation layer.
    Uses Oja's rule for unsupervised adaptation at test time.
    """

    def __init__(self, input_dim, output_dim, lr=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.lr = lr

    def forward(self, x):
        # x: (seq_len, input_dim)
        y = F.linear(x, self.weight)  # (seq_len, output_dim)
        return y

    def hebbian_update(self, x, y):
        """
        Apply Oja's rule: dW = lr * (y * x^T - y^2 * W)
        """
        # x: (seq_len, input_dim)
        # y: (seq_len, output_dim)
        with torch.no_grad():
            # Compute outer product
            outer = torch.einsum('si,so->oi', x, y)  # (output_dim, input_dim)

            # Oja's normalization term
            y_sq = (y ** 2).sum(dim=0, keepdim=True).T  # (output_dim, 1)
            norm_term = y_sq * self.weight  # (output_dim, input_dim)

            # Update weights
            self.weight.data += self.lr * (outer - norm_term)


class ARC1DModelWithHebbian(nn.Module):
    """
    ARC1D Model with Hebbian TTA layer.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128,
                 num_classes=NUM_CLASSES, hebbian_lr=0.01):
        super().__init__()

        # Build encoder and cross-attention directly (no unused base_model.decoder)
        self.encoder = SequenceEncoder(vocab_size, embed_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.hebbian = HebbianTTA(hidden_dim * 2, hidden_dim, lr=hebbian_lr)

        # Decoder after Hebbian layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

        self.hidden_dim = hidden_dim

    def encode_examples(self, support_inputs, support_outputs):
        """Encode support set examples."""
        inp_enc = self.encoder(support_inputs)
        out_enc = self.encoder(support_outputs)
        combined = torch.cat([inp_enc, out_enc], dim=1)
        example_enc = combined.mean(dim=0)
        return example_enc

    def _get_combined_features(self, inputs, example_enc_batch):
        """Get combined features for given inputs using cross-attention."""
        enc = self.encoder(inputs)
        attended, _ = self.cross_attention(enc, example_enc_batch, example_enc_batch)
        combined = torch.cat([enc, attended], dim=-1).squeeze(0)  # (seq_len, 2*hidden_dim)
        return combined

    def forward(self, support_inputs, support_outputs, query_input, adapt=False):
        """
        Forward pass with optional Hebbian adaptation.
        """
        if query_input.dim() == 1:
            query_input = query_input.unsqueeze(0)

        # Encode examples
        example_enc = self.encode_examples(support_inputs, support_outputs)
        example_enc = example_enc.unsqueeze(0)  # (1, 2*seq_len, hidden_dim)

        # Hebbian adaptation on support set (BEFORE computing query output)
        if adapt:
            for i in range(support_inputs.size(0)):
                sup_inp = support_inputs[i:i+1]
                sup_combined = self._get_combined_features(sup_inp, example_enc)
                sup_hebbian = self.hebbian(sup_combined)
                self.hebbian.hebbian_update(sup_combined, sup_hebbian)

        # Now compute query output with (possibly adapted) Hebbian weights
        query_combined = self._get_combined_features(query_input, example_enc)

        # Hebbian layer
        hebbian_out = self.hebbian(query_combined)  # (seq_len, hidden_dim)

        # Decode
        logits = self.decoder(hebbian_out)

        return logits


class MAML:
    """
    MAML training for 1D-ARC.
    Uses functional_call for proper gradient flow through the inner loop.
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, device='cuda'):
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def train_step(self, batch):
        """
        Single MAML training step on a batch of tasks.
        Uses functional_call to maintain gradient flow through the inner loop.
        """
        self.model.train()
        meta_loss = 0
        meta_acc = 0
        num_tasks = len(batch['support_inputs'])

        for task_idx in range(num_tasks):
            si = batch['support_inputs'][task_idx].to(self.device)
            so = batch['support_outputs'][task_idx].to(self.device)
            sm = batch['support_masks'][task_idx].to(self.device)
            qi = batch['query_input'][task_idx].to(self.device)
            qo = batch['query_output'][task_idx].to(self.device)
            qm = batch['query_mask'][task_idx].to(self.device)

            fast_params = {k: v.clone() for k, v in dict(self.model.named_parameters()).items()}

            # Inner loop - leave-one-out to prevent info leakage
            for step in range(self.inner_steps):
                inner_loss = 0
                for i in range(si.size(0)):
                    # Exclude example i from context
                    ctx_idx = [j for j in range(si.size(0)) if j != i]
                    if len(ctx_idx) == 0:
                        ctx_inputs = si[i:i+1]  # at least use self if only 1 example
                        ctx_outputs = so[i:i+1]
                    else:
                        ctx_inputs = si[ctx_idx]
                        ctx_outputs = so[ctx_idx]

                    logits = functional_call(self.model, fast_params,
                                             (ctx_inputs, ctx_outputs, si[i:i+1].squeeze(0)))
                    target = so[i]
                    mask = sm[i]
                    loss = F.cross_entropy(logits, target, reduction='none', ignore_index=PAD_IDX)
                    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                    inner_loss = inner_loss + loss

                inner_loss = inner_loss / si.size(0)
                grads = torch.autograd.grad(inner_loss, fast_params.values(),
                                            create_graph=True, allow_unused=True)
                fast_params = {
                    k: (v - self.inner_lr * g if g is not None else v)
                    for (k, v), g in zip(fast_params.items(), grads)
                }

            # Query with full support context (no leakage - query is separate)
            query_logits = functional_call(self.model, fast_params, (si, so, qi))
            query_loss = F.cross_entropy(query_logits, qo, reduction='none', ignore_index=PAD_IDX)
            query_loss = (query_loss * qm).sum() / qm.sum().clamp(min=1)
            meta_loss = meta_loss + query_loss

            with torch.no_grad():
                pred = query_logits.argmax(dim=-1)
                correct = ((pred == qo) * qm).sum()
                meta_acc += (correct / qm.sum().clamp(min=1)).item()

        meta_loss = meta_loss / num_tasks
        self.optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        return meta_loss.item(), meta_acc / num_tasks

    @torch.no_grad()
    def evaluate(self, dataset, num_tasks=100):
        """
        Evaluate model on a set of tasks.
        Uses functional_call for inner-loop adaptation without tracking gradients.
        """
        self.model.eval()
        correct = 0
        total = 0
        task_accs = []

        indices = random.sample(range(len(dataset)), min(num_tasks, len(dataset)))

        for idx in tqdm(indices, desc="Evaluating"):
            batch = dataset[idx]

            support_inputs = batch['support_inputs'].to(self.device)
            support_outputs = batch['support_outputs'].to(self.device)
            support_masks = batch['support_masks'].to(self.device)
            query_input = batch['query_input'].to(self.device)
            query_output = batch['query_output'].to(self.device)
            query_mask = batch['query_mask'].to(self.device)

            # Adapt on support set using functional params (no grad needed for eval)
            fast_params = {k: v.clone() for k, v in dict(self.model.named_parameters()).items()}

            for _ in range(self.inner_steps):
                inner_loss = 0
                for i in range(support_inputs.size(0)):
                    logits = functional_call(self.model, fast_params,
                                             (support_inputs, support_outputs, support_inputs[i]))
                    ce_loss = F.cross_entropy(logits, support_outputs[i], reduction='none', ignore_index=PAD_IDX)
                    inner_loss += (ce_loss * support_masks[i]).sum() / support_masks[i].sum().clamp(min=1)

                inner_loss = inner_loss / support_inputs.size(0)

                # Manual gradient computation (no graph needed for eval)
                grads = torch.autograd.grad(inner_loss, fast_params.values(), allow_unused=True)
                fast_params = {
                    k: (v - self.inner_lr * g if g is not None else v)
                    for (k, v), g in zip(fast_params.items(), grads)
                }

            # Evaluate on query
            query_logits = functional_call(self.model, fast_params,
                                           (support_inputs, support_outputs, query_input))
            predictions = query_logits.argmax(dim=-1)

            mask = query_mask.bool()
            task_correct = (predictions[mask] == query_output[mask]).sum().item()
            task_total = mask.sum().item()

            task_acc = task_correct / task_total if task_total > 0 else 0
            task_accs.append(task_acc)

            correct += task_correct
            total += task_total

        overall_acc = correct / total if total > 0 else 0
        mean_task_acc = np.mean(task_accs)

        return {
            'overall_accuracy': overall_acc,
            'mean_task_accuracy': mean_task_acc,
            'task_accuracies': task_accs
        }


def collate_fn(batch):
    """Custom collate function for batching tasks."""
    return {
        'support_inputs': [item['support_inputs'] for item in batch],
        'support_outputs': [item['support_outputs'] for item in batch],
        'support_masks': [item['support_masks'] for item in batch],
        'query_input': [item['query_input'] for item in batch],
        'query_output': [item['query_output'] for item in batch],
        'query_mask': [item['query_mask'] for item in batch],
        'task_file': [item['task_file'] for item in batch]
    }


def main():
    parser = argparse.ArgumentParser(description='1D-ARC MAML Training')
    parser.add_argument('--data_dir', type=str, default='1D-ARC/dataset', help='Path to 1D-ARC dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Meta-batch size (number of tasks)')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner loop steps')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataset
    dataset = ARC1DDataset(args.data_dir)

    # Split into train/val/test
    num_tasks = len(dataset)
    indices = list(range(num_tasks))
    random.shuffle(indices)

    train_size = int(0.7 * num_tasks)
    val_size = int(0.15 * num_tasks)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)

    # Create model
    model = ARC1DModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create MAML trainer
    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        device=device
    )

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_accs = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            loss, acc = maml.train_step(batch)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Average Acc = {avg_acc:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            print("Evaluating on validation set...")
            val_results = maml.evaluate(val_dataset, num_tasks=min(50, len(val_dataset)))
            print(f"Validation - Overall Acc: {val_results['overall_accuracy']:.4f}, "
                  f"Mean Task Acc: {val_results['mean_task_accuracy']:.4f}")

            if val_results['mean_task_accuracy'] > best_val_acc:
                best_val_acc = val_results['mean_task_accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': maml.optimizer.state_dict(),
                    'val_acc': best_val_acc
                }, os.path.join(args.save_dir, 'best_model.pt'))
                print(f"Saved best model with val acc: {best_val_acc:.4f}")

    # Reload best model before test evaluation
    best_ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_ckpt_path):
        print(f"\nReloading best model from {best_ckpt_path} (val acc: {best_val_acc:.4f})")
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_results = maml.evaluate(test_dataset, num_tasks=len(test_dataset))
    print(f"Test - Overall Acc: {test_results['overall_accuracy']:.4f}, "
          f"Mean Task Acc: {test_results['mean_task_accuracy']:.4f}")

    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'test_overall_acc': test_results['overall_accuracy'],
        'test_mean_task_acc': test_results['mean_task_accuracy'],
        'args': vars(args)
    }

    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_results['mean_task_accuracy']:.4f}")


if __name__ == '__main__':
    main()
