# Meta-Learning: Few-Shot Reasoning & Financial Prediction

**CSE559: Meta-Learning — From Few-Shot Classification to Learning to Learn**
IIIT Delhi, March 2026

10 meta-learning algorithms evaluated across 2 benchmarks with 21 experiments on 8x A100 GPUs.

## [Live Demo →](https://dk4248.github.io/Meta_Learning/)

## Benchmarks

### 1D-ARC — Abstract Reasoning ([details](META_LEARNING_1D-ARC/))
| Method | Exact Match | Token Acc |
|--------|------------|-----------|
| **Matching Networks** | **61.8%** | 92.1% |
| FOMAML | 56.6% | 91.0% |
| ANIL | 55.9% | 93.9% |
| CNP | 52.9% | 88.6% |

### Numin2 — Stock Ranking ([details](META_LEARNING_numin/))
| Method | Spearman Corr |
|--------|--------------|
| **ANIL** | **0.670** |
| Attention MAML | 0.643 |
| ProtoNet | 0.639 |
| CNP | 0.634 |

## Novel Contributions
- **Hebbian TTA + MAML** — Oja's rule online learning combined with gradient-based meta-learning
- **Stock Cross-Attention Architecture** — Per-stock LSTM + cross-stock attention within MAML
- **Task Augmentation Analysis** — Negative result: augmentation hurts financial meta-learning

## Quick Start
```bash
conda activate base
# ARC demo
jupyter notebook META_LEARNING_1D-ARC/demo_arc.ipynb
# Numin demo
jupyter notebook META_LEARNING_numin/demo_numin.ipynb
```

## Structure
```
META_LEARNING_1D-ARC/    # ARC code, models, notebook, report
META_LEARNING_numin/     # Numin code, models, notebook, report
docs/                    # GitHub Pages site
```
