"""Plot Numin2 training curves and results from experiment logs and results.json files."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re, os, json, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_val_corr(logfile, pattern="Validation"):
    vals = []
    path = os.path.join(BASE_DIR, logfile) if not os.path.isabs(logfile) else logfile
    if not os.path.exists(path):
        return vals
    with open(path) as f:
        for line in f:
            if pattern in line:
                nums = re.findall(r'[0-9]+\.[0-9]+', line)
                if nums:
                    # Try to find correlation value (usually the last float on the line)
                    vals.append(float(nums[-1]))
    return vals

def extract_train_acc(logfile):
    vals = []
    path = os.path.join(BASE_DIR, logfile) if not os.path.isabs(logfile) else logfile
    if not os.path.exists(path):
        return vals
    with open(path) as f:
        for line in f:
            m = re.search(r'Acc=([0-9.]+)', line)
            if m:
                vals.append(float(m.group(1)))
    return vals

def load_all_results():
    """Load results from all checkpoint directories."""
    numin_results = {}
    for d in sorted(glob.glob(os.path.join(BASE_DIR, "checkpoints_numin_*"))):
        rfile = os.path.join(d, 'results.json')
        if os.path.exists(rfile):
            name = os.path.basename(d).replace("checkpoints_numin_", "")
            with open(rfile) as f:
                numin_results[name] = json.load(f)
    return numin_results

def main():
    results = load_all_results()

    # Collect model data from results.json
    models = []
    val_corrs = []
    test_corrs = []
    for name, r in sorted(results.items()):
        val = r.get('best_val_corr')
        test = r.get('test_correlation')
        if isinstance(val, (int, float)) and isinstance(test, (int, float)):
            models.append(name)
            val_corrs.append(val)
            test_corrs.append(test)

    if not models:
        print("No Numin results found. Run experiments first.")
        return

    # Sort by test correlation descending
    sorted_idx = sorted(range(len(models)), key=lambda i: test_corrs[i], reverse=True)
    models = [models[i] for i in sorted_idx]
    val_corrs = [val_corrs[i] for i in sorted_idx]
    test_corrs = [test_corrs[i] for i in sorted_idx]

    # Figure 1: 4-panel training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Numin2 Meta-Learning: Training Curves', fontsize=16, fontweight='bold')

    # Plot 1: Best models - Val Correlation
    ax = axes[0][0]
    log_configs = [
        ('logs/numin_reptile.log', 'reptile', 'b-o'),
        ('logs/numin_reptile_v2.log', 'reptile_v2', 'm-d'),
        ('logs/numin_fomaml.log', 'fomaml', 'g-^'),
        ('logs/numin_protonet.log', 'protonet', 'r-s'),
        ('logs/numin_ensemble.log', 'ensemble', 'c-*'),
    ]
    for logfile, label, style in log_configs:
        vals = extract_val_corr(logfile)
        if vals:
            test_val = results.get(label, {}).get('test_correlation', '?')
            test_str = f"{test_val:.3f}" if isinstance(test_val, float) else str(test_val)
            ep = np.arange(5, 5*len(vals)+1, 5)[:len(vals)]
            ax.plot(ep, vals, style, label=f'{label} (Test: {test_str})', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Spearman Correlation')
    ax.set_title('Validation Correlation Over Training')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: MAML-family comparison
    ax = axes[0][1]
    log_configs2 = [
        ('logs/numin_maml.log', 'maml', 'b-o'),
        ('logs/numin_anil.log', 'anil', 'r-s'),
        ('logs/numin_attention.log', 'attention', 'g-^'),
        ('logs/numin_transformer.log', 'transformer', 'c-d'),
    ]
    for logfile, label, style in log_configs2:
        vals = extract_val_corr(logfile)
        if vals:
            test_val = results.get(label, {}).get('test_correlation', '?')
            test_str = f"{test_val:.3f}" if isinstance(test_val, float) else str(test_val)
            ep = np.arange(5, 5*len(vals)+1, 5)[:len(vals)]
            ax.plot(ep, vals, style, label=f'{label} (Test: {test_str})', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Spearman Correlation')
    ax.set_title('MAML-Family Variants')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 3: Training accuracy
    ax = axes[1][0]
    reptile_acc = extract_train_acc('logs/numin_reptile.log')
    if reptile_acc:
        ax.plot(range(1, len(reptile_acc)+1), reptile_acc, 'b-', linewidth=2, label='Reptile Train Acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Accuracy')
    ax.set_title('Reptile: Training Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 4: Final Results Bar Chart (data-driven)
    ax = axes[1][1]
    display_models = [m.replace('_', '\n') for m in models[:10]]
    display_val = val_corrs[:10]
    display_test = test_corrs[:10]

    x = np.arange(len(display_models))
    width = 0.35
    ax.bar(x - width/2, display_val, width, label='Validation', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, display_test, width, label='Test', color='coral', alpha=0.8)
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('All Models: Validation vs Test Correlation')
    ax.set_xticks(x); ax.set_xticklabels(display_models, fontsize=7)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    outpath = os.path.join(BASE_DIR, 'numin2_training_curves.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")

    # Figure 2: Method taxonomy
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

    families = {
        'Optimization-based\n(Reptile/FOMAML)': [],
        'Gradient-based\n(MAML/ANIL)': [],
        'Metric-based\n(ProtoNet)': [],
        'Model-based\n(Attention/CNP)': [],
    }
    for name, r in results.items():
        test = r.get('test_correlation')
        if not isinstance(test, (int, float)):
            continue
        nl = name.lower()
        if 'reptile' in nl or 'fomaml' in nl or 'ensemble' in nl or 'augmented' in nl or 'aggressive' in nl:
            families['Optimization-based\n(Reptile/FOMAML)'].append(test)
        elif 'maml' in nl or 'anil' in nl:
            families['Gradient-based\n(MAML/ANIL)'].append(test)
        elif 'proto' in nl:
            families['Metric-based\n(ProtoNet)'].append(test)
        else:
            families['Model-based\n(Attention/CNP)'].append(test)

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
        ax2.set_ylabel('Best Test Spearman Correlation', fontsize=12)
        ax2.set_title('Meta-Learning Families on Numin2', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 0.8)
        for bar, val in zip(bars, best_tests):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    outpath2 = os.path.join(BASE_DIR, 'numin2_taxonomy.png')
    plt.savefig(outpath2, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath2}")

    print("\nAll plots generated successfully!")

if __name__ == '__main__':
    main()
