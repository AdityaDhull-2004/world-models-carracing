import os

HOME       = os.path.expanduser("~")
EXP_DIR    = os.path.join(HOME, "world-models-dinov2-v2", "experiments")

ROLLOUT_DIR    = os.path.join(EXP_DIR, "random-rollouts")
LATENT_DIR     = os.path.join(EXP_DIR, "latent-stats")
MEMORY_DIR     = os.path.join(EXP_DIR, "memory-training")
CONTROLLER_DIR = os.path.join(EXP_DIR, "controller")

for d in [LATENT_DIR, MEMORY_DIR, CONTROLLER_DIR]:
    os.makedirs(d, exist_ok=True)

# DINOv2 — 98x98 gives 7x7=49 patches (best balance of quality vs speed)
# This is the resolution that gave best score of 52.1 in previous run
DINOV2_WEIGHTS  = os.path.join(HOME, "world-models-dinov2-v2",
                                "pretrained", "dinov2_vits14.pth")
DINOV2_IMG_SIZE = 98      # 98x98 -> 7x7=49 patches (14*7=98)
Z_DIM           = 384     # ViT-Small CLS token dimension (fixed)

IMAGE_SIZE   = 64
ACTION_DIM   = 3
LSTM_UNITS   = 256
NUM_MIXTURES = 5
SEQ_LEN      = 500

# RNN — 10 epochs with high learning rate to avoid flat loss
RNN_BATCH  = 32
RNN_EPOCHS = 10

# CMA-ES — same as best run that gave 52.1
# sigma=0.3 prevents local optima (0.1 caused plateau in earlier runs)
POP_SIZE           = 32
ROLLOUTS_PER_AGENT = 4
MAX_EPISODE_STEPS  = 400
MAX_GENERATIONS    = 150
