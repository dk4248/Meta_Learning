"""Per-task-type analysis of 1D-ARC results using best model checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json, os, glob, random, argparse
import numpy as np
from copy import deepcopy
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Reuse model and dataset from arc_reptile
import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
from arc_reptile import ARC1DDataset, ARC1DModel

def evaluate_per_type(model_path, dataset, test_indices, inner_lr=0.01, inner_steps=10, device='cuda'):
    model = ARC1DModel()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    type_results = defaultdict(list)

    for idx in test_indices:
        task = dataset[idx]
        task_file = dataset.tasks[idx]
        task_type = os.path.basename(os.path.dirname(task_file))

        support_inputs = task['support_inputs'].to(device)
        support_outputs = task['support_outputs'].to(device)
        support_masks = task['support_masks'].to(device)
        query_input = task['query_input'].to(device)
        query_output = task['query_output'].to(device)
        query_mask = task['query_mask'].to(device)

        # Save and adapt
        old_weights = {n: p.data.clone() for n, p in model.named_parameters()}
        inner_opt = optim.SGD(model.parameters(), lr=inner_lr)
        model.train()

        K = support_inputs.size(0)
        for _ in range(inner_steps):
            loss = 0
            for i in range(K):
                # Leave-one-out: exclude example i from context
                ctx_idx = [j for j in range(K) if j != i]
                if len(ctx_idx) > 0:
                    ctx_in = support_inputs[ctx_idx]
                    ctx_out = support_outputs[ctx_idx]
                else:
                    ctx_in = support_inputs[i:i+1]
                    ctx_out = support_outputs[i:i+1]
                logits = model(ctx_in, ctx_out, support_inputs[i])
                ce = F.cross_entropy(logits, support_outputs[i], reduction='none')
                loss += (ce * support_masks[i]).sum() / support_masks[i].sum().clamp(min=1)
            loss /= K
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            inner_opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(support_inputs, support_outputs, query_input)
            preds = logits.argmax(dim=-1)
            mask = query_mask.bool()
            acc = (preds[mask] == query_output[mask]).float().mean().item()
            type_results[task_type].append(acc)

        # Restore
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data = old_weights[n]

    return type_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints_arc_reptile/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='1D-ARC/dataset')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    model_path = os.path.join(base_dir, args.model_path) if not os.path.isabs(args.model_path) else args.model_path

    dataset = ARC1DDataset(data_dir)
    indices = list(range(len(dataset)))

    # Chronological split matching training script
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    test_idx = indices[train_size + val_size:]

    print(f"Evaluating {model_path} per task type...")
    print(f"Test set size: {len(test_idx)} tasks")

    results = evaluate_per_type(
        model_path, dataset, test_idx,
        inner_lr=0.01, inner_steps=10, device=device
    )

    print("\n" + "="*70)
    print("PER-TASK-TYPE ANALYSIS")
    print("="*70)
    print(f"{'Task Type':<25} {'Acc':>8} {'Count':>6} {'Std':>8}")
    print("-"*50)

    all_types = sorted(results.keys())
    type_data = []
    for t in all_types:
        accs = results[t]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        type_data.append((t, mean_acc, len(accs), std_acc))
        print(f"{t:<25} {mean_acc:>8.4f} {len(accs):>6} {std_acc:>8.4f}")

    type_data.sort(key=lambda x: x[1], reverse=True)
    print("\n--- SORTED BY ACCURACY ---")
    for t, acc, cnt, std in type_data:
        bar = "█" * int(acc * 30)
        print(f"{t:<25} {acc:>6.1%} {bar}")

    overall = np.mean([acc for accs in results.values() for acc in accs])
    print(f"\nOverall: {overall:.4f}")

    # Save for report
    output_path = os.path.join(base_dir, 'per_task_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({t: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'count': len(v)}
                   for t, v in results.items()}, f, indent=2)
    print(f"\nSaved {output_path}")

if __name__ == '__main__':
    main()
