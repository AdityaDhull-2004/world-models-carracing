"""Encode all rollout frames with the trained VAE → mu, logvar per frame."""
import os, glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from worldmodels.config import ROLLOUT_DIR, LATENT_DIR
from worldmodels.models.vae import ConvVAE


def main():
    vae   = ConvVAE.load_from()
    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.npz")))
    assert len(files) > 0, f"No rollouts found in {ROLLOUT_DIR}"
    print(f"Encoding {len(files)} rollouts...")

    for fpath in tqdm(files):
        data    = np.load(fpath)
        obs     = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        mus, logvars = [], []
        for i in range(0, len(obs), 64):
            mu, logvar = vae.encode(tf.constant(obs[i:i+64]))
            mus.append(mu.numpy())
            logvars.append(logvar.numpy())
        out = os.path.join(LATENT_DIR, os.path.basename(fpath))
        np.savez_compressed(out,
                            mu=np.concatenate(mus),
                            logvar=np.concatenate(logvars),
                            actions=actions)

    print(f"Done. Latent stats saved to: {LATENT_DIR}")

if __name__ == "__main__":
    main()