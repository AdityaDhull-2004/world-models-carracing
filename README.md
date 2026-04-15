# World Models — CarRacing-v2 Experiments

Comparing vision models in the World Models framework (Ha & Schmidhuber, 2018).

## Results Summary

| Experiment | Vision Model | Resolution | Score | Training Time |
|---|---|---|---|---|
| Exp 1: ConvVAE | ConvVAE (from scratch) | 64x64 | **339.3 ± 86.4** | ~45 hours |
| Exp 2: DINOv2 v1 | DINOv2 ViT-S/14 | 70x70 | -33.2 ± 0.5 | ~12 hours |
| Exp 3: DINOv2 v2 | DINOv2 ViT-S/14 | 98x98 | 157.62 (training best) | ~23 hours |

## Structure
- `world-models-vae/` — Experiment 1: ConvVAE
- `world-models-dinov2/` — Experiment 2: DINOv2 v1 (70x70)
- `world-models-dinov2-v2/` — Experiment 3: DINOv2 v2 (98x98, fixed RNN)

## Key Findings
- ConvVAE significantly outperforms DINOv2 in this framework
- DINOv2 v1 failed because RNN loss was flat (lr=1e-4 too small for 384-dim)
- DINOv2 v2 improved by using lr=1e-3 — best training reward reached 157.62
- DINOv2 v2 CMA-ES converged early due to small sigma=0.05

## Based on
Ha, D. & Schmidhuber, J. (2018). World Models. arXiv:1803.10122
