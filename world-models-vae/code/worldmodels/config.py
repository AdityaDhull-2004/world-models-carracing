import os

HOME       = os.path.expanduser("~")
EXP_DIR    = os.path.join(HOME, "world-models", "experiments")

ROLLOUT_DIR    = os.path.join(EXP_DIR, "random-rollouts")
LATENT_DIR     = os.path.join(EXP_DIR, "latent-stats")
VAE_DIR        = os.path.join(EXP_DIR, "vae-training")
MEMORY_DIR     = os.path.join(EXP_DIR, "memory-training")
CONTROLLER_DIR = os.path.join(EXP_DIR, "controller")

for d in [ROLLOUT_DIR, LATENT_DIR, VAE_DIR, MEMORY_DIR, CONTROLLER_DIR]:
    os.makedirs(d, exist_ok=True)

# Model dimensions
IMAGE_SIZE   = 64
Z_DIM        = 32
ACTION_DIM   = 3
LSTM_UNITS   = 256
NUM_MIXTURES = 5
SEQ_LEN      = 500

# Training
VAE_BATCH  = 64
VAE_EPOCHS = 10
RNN_BATCH  = 32
RNN_EPOCHS = 80

# CMA-ES (Controller)
POP_SIZE           = 32
ROLLOUTS_PER_AGENT = 4
MAX_EPISODE_STEPS  = 400
MAX_GENERATIONS    = 500