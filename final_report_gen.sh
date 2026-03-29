#!/bin/bash
# Wait for all experiments to complete, then generate final reports
cd "$(dirname "$0")"

echo "Waiting for all experiments to complete..."
while true; do
    running=$(ps aux | grep "python.*\(arc_\|numin_\)" | grep -v grep | wc -l)
    if [ "$running" -eq 0 ]; then
        break
    fi
    echo "$(date +%H:%M:%S): $running experiments still running..."
    sleep 60
done

echo ""
echo "All experiments complete! Generating final outputs..."
echo ""

# Regenerate plots
python plot_arc_curves.py
python plot_numin_curves.py

# Generate report
python generate_report.py | tee final_report_output.txt

# Per-task analysis (if reptile checkpoint exists)
if [ -f "checkpoints_arc_reptile/best_model.pt" ]; then
    CUDA_VISIBLE_DEVICES=0 python per_task_analysis.py --gpu 0
fi

# Generate per-task analysis plot
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os

base = os.path.dirname(os.path.abspath('$0')) or '.'
pf = os.path.join(base, 'per_task_analysis.json')
if not os.path.exists(pf):
    print('No per_task_analysis.json found')
    exit()

with open(pf) as f:
    data = json.load(f)

types = sorted(data.keys(), key=lambda t: data[t]['mean'], reverse=True)
means = [data[t]['mean'] for t in types]
stds = [data[t]['std'] for t in types]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
colors = ['#2ecc71' if m > 0.9 else '#f39c12' if m > 0.7 else '#e74c3c' for m in means]
bars = ax.barh(range(len(types)), means, xerr=stds, color=colors, alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(types)))
ax.set_yticklabels([t.replace('1d_', '') for t in types], fontsize=9)
ax.set_xlabel('Accuracy')
ax.set_title('Per-Task-Type Accuracy (Best Model)', fontweight='bold')
ax.set_xlim(0, 1.0)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(base, 'per_task_analysis.png'), dpi=150, bbox_inches='tight')
print('Saved: per_task_analysis.png')
"

# Generate combined final results plot
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, glob

base = os.path.dirname(os.path.abspath('$0')) or '.'

# Collect all results
numin_results = {}
arc_results = {}
for d in sorted(glob.glob(os.path.join(base, 'checkpoints_*'))):
    rf = os.path.join(d, 'results.json')
    if not os.path.exists(rf): continue
    with open(rf) as f:
        data = json.load(f)
    name = os.path.basename(d).replace('checkpoints_', '')
    if 'arc' in name:
        test = data.get('test_accuracy', data.get('test_mean_task_acc'))
        if isinstance(test, float):
            arc_results[name.replace('arc_', '')] = test
    else:
        test = data.get('test_correlation')
        if isinstance(test, float):
            numin_results[name.replace('numin_', '').replace('numin', 'maml')] = test

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Meta-Learning Results: All Methods', fontsize=16, fontweight='bold')

# Numin
if numin_results:
    names = sorted(numin_results.keys(), key=lambda k: numin_results[k], reverse=True)
    vals = [numin_results[n] for n in names]
    colors = ['#2ecc71' if v > 0.6 else '#f39c12' if v > 0.5 else '#e74c3c' for v in vals]
    bars = ax1.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='black')
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names)
    ax1.set_xlabel('Test Spearman Correlation'); ax1.set_title('Numin2 Benchmark')
    for i, v in enumerate(vals): ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    ax1.set_xlim(0, max(vals) * 1.15)
    ax1.invert_yaxis(); ax1.grid(True, alpha=0.3, axis='x')

# ARC
if arc_results:
    names = sorted(arc_results.keys(), key=lambda k: arc_results[k], reverse=True)
    vals = [arc_results[n] for n in names]
    colors = ['#2ecc71' if v > 0.85 else '#f39c12' if v > 0.7 else '#e74c3c' for v in vals]
    bars = ax2.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='black')
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names)
    ax2.set_xlabel('Test Accuracy'); ax2.set_title('1D-ARC Benchmark')
    for i, v in enumerate(vals): ax2.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=9)
    ax2.set_xlim(0, min(max(vals) * 1.15, 1.0))
    ax2.invert_yaxis(); ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(base, 'final_all_results.png'), dpi=150, bbox_inches='tight')
print('Saved: final_all_results.png')
"

echo "DONE! All reports generated."
