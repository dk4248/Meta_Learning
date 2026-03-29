"""Plot 1D-ARC training curves and results from experiment logs and results.json files."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re, os, json, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_vals(logfile, pattern, idx=0):
    vals = []
    path = os.path.join(BASE_DIR, logfile) if not os.path.isabs(logfile) else logfile
    if not os.path.exists(path):
        return vals
    with open(path) as f:
        for line in f:
            if pattern in line:
                nums = re.findall(r'[0-9]+\.[0-9]+', line)
                if nums and idx < len(nums):
                    vals.append(float(nums[idx]))
    return vals

def load_all_results():
    """Load results from all checkpoint directories."""
    arc_results = {}
    for d in sorted(glob.glob(os.path.join(BASE_DIR, "checkpoints_arc_*"))):
        rfile = os.path.join(d, 'results.json')
        if os.path.exists(rfile):
            name = os.path.basename(d).replace("checkpoints_arc_", "")
            with open(rfile) as f:
                arc_results[name] = json.load(f)
    return arc_results

def main():
    results = load_all_results()

    # Collect model names, val accuracies, test accuracies from results.json
    models = []
    val_accs = []
    test_accs = []
    for name, r in sorted(results.items()):
        val = r.get('best_val_acc', r.get('best_val_accuracy'))
        test = r.get('test_accuracy', r.get('test_mean_task_acc'))
        if isinstance(val, (int, float)) and isinstance(test, (int, float)):
            models.append(name)
            val_accs.append(val)
            test_accs.append(test)

    if not models:
        print("No ARC results found. Run experiments first.")
        return

    # Sort by test accuracy descending
    sorted_idx = sorted(range(len(models)), key=lambda i: test_accs[i], reverse=True)
    models = [models[i] for i in sorted_idx]
    val_accs = [val_accs[i] for i in sorted_idx]
    test_accs = [test_accs[i] for i in sorted_idx]

    # Figure 1: 4-panel training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1D-ARC Meta-Learning: Training Curves', fontsize=16, fontweight='bold')

    # Plot 1: Val accuracy from logs
    ax = axes[0][0]
    log_configs = [
        ('logs/arc_reptile.log', 'Validation', 'reptile', 'b-o'),
        ('logs/arc_reptile_is20.log', 'Validation', 'reptile_is20', 'c-s'),
        ('logs/arc_fomaml.log', 'Validation', 'fomaml', 'g-^'),
        ('logs/arc_protonet.log', 'Validation', 'protonet', 'r-d'),
    ]
    for logfile, pattern, label, style in log_configs:
        vals = extract_vals(logfile, pattern)
        if vals:
            test_val = results.get(label, {}).get('test_accuracy', results.get(label, {}).get('test_mean_task_acc', '?'))
            test_str = f"{test_val:.1%}" if isinstance(test_val, float) else str(test_val)
            ep = np.arange(5, 5*len(vals)+1, 5)[:len(vals)]
            ax.plot(ep, vals, style, label=f'{label} (Test: {test_str})', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Over Training')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: More variants
    ax = axes[0][1]
    log_configs2 = [
        ('logs/arc_anil.log', 'Validation', 'anil', 'm', 'o'),
        ('logs/arc_cnp.log', 'Validation', 'cnp', 'c', 's'),
        ('logs/arc_matching.log', 'Validation', 'matching', 'orange', '^'),
        ('logs/arc_transformer_maml.log', 'Validation', 'transformer', 'brown', 'd'),
    ]
    for cfg in log_configs2:
        logfile, pattern, label, color, marker = cfg
        vals = extract_vals(logfile, pattern)
        if vals:
            test_val = results.get(label, {}).get('test_accuracy', results.get(label, {}).get('test_mean_task_acc', '?'))
            test_str = f"{test_val:.1%}" if isinstance(test_val, float) else str(test_val)
            ep = np.arange(5, 5*len(vals)+1, 5)[:len(vals)]
            ax.plot(ep, vals, color=color, marker=marker, label=f'{label} (Test: {test_str})', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy')
    ax.set_title('Other Meta-Learning Variants')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 3: Training accuracy from best model's log
    ax = axes[1][0]
    train_acc = extract_vals('logs/arc_reptile.log', 'Acc=')
    if not train_acc:
        # Try extracting with different pattern
        train_acc = []
        logpath = os.path.join(BASE_DIR, 'logs/arc_reptile.log')
        if os.path.exists(logpath):
            with open(logpath) as f:
                for line in f:
                    m = re.search(r'Acc=([0-9.]+)', line)
                    if m: train_acc.append(float(m.group(1)))
    if train_acc:
        ax.plot(range(1, len(train_acc)+1), train_acc, 'b-', linewidth=1.5, label='Reptile Train Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Accuracy')
    ax.set_title('Reptile: Training Accuracy on 1D-ARC')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 4: Final bar chart from results.json (data-driven)
    ax = axes[1][1]
    display_models = [m.replace('_', '\n') for m in models[:8]]  # top 8
    display_val = val_accs[:8]
    display_test = test_accs[:8]

    x = np.arange(len(display_models))
    width = 0.35
    ax.bar(x - width/2, display_val, width, label='Validation', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, display_test, width, label='Test', color='coral', alpha=0.8)
    ax.set_ylabel('Accuracy'); ax.set_title('All Models: Validation vs Test Accuracy')
    ax.set_xticks(x); ax.set_xticklabels(display_models, fontsize=7)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath = os.path.join(BASE_DIR, 'arc_training_curves.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")

    # Figure 2: Taxonomy bar chart
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    # Group models by family
    families = {
        'Optimization-based\n(Reptile/FOMAML)': [],
        'Gradient-based\n(MAML/ANIL)': [],
        'Metric-based\n(ProtoNet/Matching)': [],
        'Model-based\n(CNP/Transformer)': [],
    }
    for name, r in results.items():
        test = r.get('test_accuracy', r.get('test_mean_task_acc'))
        if not isinstance(test, (int, float)):
            continue
        nl = name.lower()
        if 'reptile' in nl or 'fomaml' in nl:
            families['Optimization-based\n(Reptile/FOMAML)'].append(test)
        elif 'maml' in nl or 'anil' in nl:
            families['Gradient-based\n(MAML/ANIL)'].append(test)
        elif 'proto' in nl or 'matching' in nl:
            families['Metric-based\n(ProtoNet/Matching)'].append(test)
        else:
            families['Model-based\n(CNP/Transformer)'].append(test)

    cats = []
    best_tests = []
    colors_map = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    for cat, vals in families.items():
        if vals:
            cats.append(cat)
            best_tests.append(max(vals))

    if cats:
        colors = colors_map[:len(cats)]
        bars = ax2.bar(cats, best_tests, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Best Test Accuracy', fontsize=12)
        ax2.set_title('Meta-Learning Families on 1D-ARC', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        for bar, val in zip(bars, best_tests):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.1%}', ha='center', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath2 = os.path.join(BASE_DIR, 'arc_taxonomy.png')
    plt.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath2}")

if __name__ == '__main__':
    main()
