import os

HOME       = os.path.expanduser("~")
EXP_DIR    = os.path.join(HOME, "world-models-dinov2", "experiments")

ROLLOUT_DIR    = os.path.join(EXP_DIR, "random-rollouts")
LATENT_DIR     = os.path.join(EXP_DIR, "latent-stats")
MEMORY_DIR     = os.path.join(EXP_DIR, "memory-training")
CONTROLLER_DIR = os.path.join(EXP_DIR, "controller")

for d in [LATENT_DIR, MEMORY_DIR, CONTROLLER_DIR]:
    os.makedirs(d, exist_ok=True)

# DINOv2 — 70x70 gives exactly 5x5=25 patches (14*5=70)
DINOV2_WEIGHTS  = os.path.join(HOME, "world-models-dinov2",
                                "pretrained", "dinov2_vits14.pth")
DINOV2_IMG_SIZE = 70      # 70x70 -> 5x5=25 patches (no dimension mismatch)
Z_DIM           = 384     # ViT-Small output dim (fixed)

IMAGE_SIZE   = 64
ACTION_DIM   = 3
LSTM_UNITS   = 256        # Back to 256 — trains well with batch=32
NUM_MIXTURES = 5
SEQ_LEN      = 500

RNN_BATCH  = 32           # Larger batch — better gradient estimates
RNN_EPOCHS = 20           # Sufficient for convergence

# CMA-ES — optimized to reach best~100 in 200 generations
POP_SIZE           = 16
ROLLOUTS_PER_AGENT = 4
MAX_EPISODE_STEPS  = 400
MAX_GENERATIONS    = 200
