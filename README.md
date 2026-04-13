# World Models — CarRacing-v2 Experiments

Comparing 2 vision models in the World Models framework (Ha & Schmidhuber, 2018).

| Experiment | Vision Model | Score | Training Time |
|---|---|---|---|
| Exp 1 | ConvVAE (from scratch) | **339.3 ± 86.4** | ~45 hours |
| Exp 2 | DINOv2 ViT-Small/14 | -33.2 ± 0.5 | ~12 hours |

## Structure
- `world-models-vae/` — ConvVAE experiment
- `world-models-dinov2/` — DINOv2 experiment

## Based on
Ha, D. & Schmidhuber, J. (2018). World Models. arXiv:1803.10122
