# Meta-Learning for 1D-ARC Abstract Reasoning

7 meta-learning algorithms evaluated on the 1D Abstraction and Reasoning Corpus (901 tasks, 18 transformation types).

## Results (Exact Match on Test Set)

| Rank | Method | EM (%) | Token Acc (%) |
|------|--------|--------|---------------|
| 1 | **Matching Networks** | **61.8** | 92.1 |
| 2 | FOMAML | 56.6 | 91.0 |
| 3 | ANIL | 55.9 | 93.9 |
| 4 | CNP | 52.9 | 88.6 |
| 5 | Reptile (is10) | 38.2 | 87.8 |
| 6 | Reptile (is20) | 35.3 | 87.5 |
| 7 | ProtoNet | 0.0 | 53.4 |

## Novel Contribution
- **Hebbian Test-Time Adaptation + MAML**: Oja's rule online learning combined with gradient-based meta-learning

## Quick Start
```bash
conda activate base
jupyter notebook demo_arc.ipynb
```

## Structure
```
arc_*.py          # 7 experiment scripts (one per algorithm)
demo_arc.ipynb    # Interactive demo notebook
models/           # Trained model checkpoints
results/          # EM and per-task-type results
plots/            # Generated figures
docs/             # LaTeX report and PDF
1D-ARC/           # Dataset (901 JSON task files)
```

## Run Experiments
```bash
CUDA_VISIBLE_DEVICES=0 python arc_reptile.py --data_dir 1D-ARC/dataset --epochs 100 --gpu 0
CUDA_VISIBLE_DEVICES=1 python arc_matching.py --data_dir 1D-ARC/dataset --epochs 100 --gpu 0
```
