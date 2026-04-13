# World Models — Experiment 1: ConvVAE

## Final Results
- **Mean Score: 339.3 ± 86.4** (100 episodes)
- Best single episode: 512.08
- Paper target: 906 ± 21

## Architecture
- Vision (V): ConvVAE trained from scratch, Z_DIM=32, kl_weight=0.01
- Memory (M): MDN-RNN, LSTM-256, 5 Gaussian mixtures
- Controller (C): Linear policy, 867 parameters, CMA-ES optimizer

## Config
- VAE: 10 epochs, batch=64, kl_weight=0.01
- RNN: 80 epochs, batch=32, SEQ_LEN=500
- CMA-ES: POP=32, ROLLOUTS=4, MAX_STEPS=400, 500 generations, sigma=0.1

## Hardware
- NVIDIA RTX 3050 (6GB VRAM), Ubuntu 22.04
- Total training time: ~45 hours
