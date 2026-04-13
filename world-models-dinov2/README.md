# World Models — Experiment 2: DINOv2 ViT-Small/14

## Final Results
- **Mean Score: -33.2 ± 0.5** (100 episodes)
- Best single episode: -31.74
- Training best reward: 30.15 at generation 183

## Architecture
- Vision (V): DINOv2 ViT-Small/14 pretrained on LVD-142M (142M images)
- Input: 70x70 pixels → 5x5 = 25 patches
- Z_DIM: 384 (vs 32 for ConvVAE)
- Memory (M): MDN-RNN, LSTM-256, 5 Gaussian mixtures
- Controller (C): Linear policy, 1923 parameters, CMA-ES optimizer

## Config
- No vision training needed (pretrained)
- RNN: 20 epochs, batch=32, SEQ_LEN=500
- CMA-ES: POP=16, ROLLOUTS=4, MAX_STEPS=400, 200 generations, sigma=0.3

## Hardware
- NVIDIA RTX 3050 (6GB VRAM), Ubuntu 22.04
- Total training time: ~12 hours
