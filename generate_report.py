"""
Generate final report for Meta-Learning Course Project

Collects results from all experiments and generates a summary report.
"""

import json
import os
import glob
from datetime import datetime

def load_results(checkpoint_dir):
    """Load results from a checkpoint directory."""
    results_file = os.path.join(checkpoint_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def safe_format(val, fmt='.4f'):
    """Safely format a value, handling N/A strings and NaN."""
    if isinstance(val, (int, float)):
        if val != val:  # NaN check
            return 'N/A'
        return f"{val:{fmt}}"
    return str(val)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("META-LEARNING COURSE PROJECT - FINAL REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Numin2 Results
    print("\n" + "=" * 70)
    print("NUMIN2 BENCHMARK - Stock Rank Prediction")
    print("=" * 70)
    print("\nTask: Predict relative rankings of 50 Nifty-50 stocks")
    print("Metric: Spearman Correlation")
    print("\nResults:")
    print("-" * 50)

    numin_dirs = sorted(glob.glob(os.path.join(base_dir, "checkpoints_numin_*")))
    numin_results = []

    for d in numin_dirs:
        results = load_results(d)
        if results:
            name = os.path.basename(d).replace("checkpoints_numin_", "")
            val_corr = results.get('best_val_corr', 'N/A')
            test_corr = results.get('test_correlation', 'N/A')

            if isinstance(val_corr, float) and val_corr == val_corr:  # not NaN
                numin_results.append({
                    'name': name,
                    'val_corr': val_corr,
                    'test_corr': test_corr,
                    'args': results.get('args', {})
                })

    numin_results.sort(
        key=lambda x: x['test_corr'] if isinstance(x['test_corr'], float) and x['test_corr'] == x['test_corr'] else -999,
        reverse=True
    )

    print(f"{'Model':<25} {'Val Corr':>12} {'Test Corr':>12}")
    print("-" * 50)
    for r in numin_results:
        val = safe_format(r['val_corr'])
        test = safe_format(r['test_corr'])
        print(f"{r['name']:<25} {val:>12} {test:>12}")

    if numin_results:
        best = numin_results[0]
        print(f"\nBest Model: {best['name']}")
        print(f"  Test Correlation: {safe_format(best['test_corr'])}")

    # 1D-ARC Results
    print("\n" + "=" * 70)
    print("1D-ARC BENCHMARK - Abstract Reasoning")
    print("=" * 70)
    print("\nTask: Few-shot 1D sequence transformation learning")
    print("Metric: Token-level Accuracy")
    print("\nResults:")
    print("-" * 50)

    arc_dirs = sorted(glob.glob(os.path.join(base_dir, "checkpoints_arc_*")))
    arc_results = []

    for d in arc_dirs:
        results = load_results(d)
        if results:
            name = os.path.basename(d).replace("checkpoints_arc_", "")
            val_acc = results.get('best_val_acc', results.get('best_val_accuracy', 'N/A'))
            test_acc = results.get('test_accuracy', results.get('test_mean_task_acc', 'N/A'))

            if isinstance(val_acc, float):
                arc_results.append({
                    'name': name,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'args': results.get('args', {})
                })

    arc_results.sort(key=lambda x: x['test_acc'] if isinstance(x['test_acc'], float) else -1, reverse=True)

    print(f"{'Model':<25} {'Val Acc':>12} {'Test Acc':>12}")
    print("-" * 50)
    for r in arc_results:
        val = safe_format(r['val_acc'])
        test = safe_format(r['test_acc'])
        print(f"{r['name']:<25} {val:>12} {test:>12}")

    if arc_results:
        best = arc_results[0]
        print(f"\nBest Model: {best['name']}")
        print(f"  Test Accuracy: {safe_format(best['test_acc'])}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Findings:")
    if numin_results:
        best = numin_results[0]
        print(f"- NUMIN2: {best['name']} achieves {safe_format(best['test_corr'])} test correlation")
    if arc_results:
        best = arc_results[0]
        print(f"- 1D-ARC: {best['name']} achieves {safe_format(best['test_acc'])} test accuracy")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
