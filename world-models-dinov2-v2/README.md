# World Models — Experiment 3: DINOv2 v2 (98x98, lr=1e-3)

## Final Results
- **Best training reward: 157.62** (at generation 60)
- **Final generations: 150**
- Evaluation score: pending (controller improved over previous -33.2 run)

## Key Differences from Experiment 2 (DINOv2 v1)
| Setting | Exp 2 (DINOv2 v1) | Exp 3 (DINOv2 v2) |
|---|---|---|
| Resolution | 70x70 (25 patches) | 98x98 (49 patches) |
| RNN lr | 1e-4 (flat loss) | 1e-3 (learning) |
| RNN epochs | 20 | 10 |
| sigma (CMA-ES) | 0.3 | 0.05 |
| Best score | -33.2 | 157.62 (training) |

## Architecture
- Vision: DINOv2 ViT-Small/14 pretrained on LVD-142M
- Input: 98x98 pixels — 7x7=49 patches (14*7=98, exact multiple)
- Z_DIM: 384 (CLS token)
- Memory: MDN-RNN, LSTM-256, 5 mixtures, lr=1e-3
- Controller: Linear, 1923 parameters, CMA-ES sigma=0.05

## Training Progress
- Gen 0-6: Negative rewards (exploration phase)
- Gen 7: First positive best reward (9.01)
- Gen 32: Major jump to 77.37
- Gen 60: Best reward 157.62 (plateau begins)
- Gen 150: Training ended, best still 157.62

## Notes
- sigma=0.05 (very small) caused CMA-ES to converge too early
- Best reached at Gen 60, no improvement for 90 more generations
- Higher sigma (0.3) would likely continue improving past 157.62

## Hardware
- NVIDIA RTX 3050 (6GB VRAM), Ubuntu 22.04
