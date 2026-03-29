# Meta-Learning for Stock Rank Prediction (Numin2)

11 meta-learning algorithms for predicting relative rankings of 50 Nifty-50 stocks.

## Results (Spearman Correlation on Test Set)

| Rank | Method | Test Corr |
|------|--------|-----------|
| 1 | **ANIL** | **0.670** |
| 2 | Attention MAML | 0.643 |
| 3 | ProtoNet | 0.639 |
| 4 | CNP | 0.634 |
| 5 | FOMAML | 0.631 |
| 6 | Aggressive Reptile | 0.628 |
| 7 | Ensemble (LSTM+T) | 0.623 |
| 8 | Transformer MAML | 0.622 |
| 9 | Reptile | 0.614 |
| 10 | Augmented Reptile | 0.612 |
| 11 | MAML | 0.565 |

## Novel Contributions
- **Stock Cross-Attention MAML**: Per-stock LSTM + cross-stock attention architecture
- **Task-level augmentation analysis**: Negative result showing augmentation hurts financial meta-learning

## Quick Start
```bash
conda activate base
jupyter notebook demo_numin.ipynb
```

## Structure
```
numin_*.py          # 12 experiment scripts
demo_numin.ipynb    # Interactive demo notebook
models/             # Trained model checkpoints
results/            # Per-model results
plots/              # Generated figures
docs/               # LaTeX report and PDF
numin_sample.parquet # Dataset
```
