import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from worldmodels.config import ROLLOUT_DIR, VAE_DIR, VAE_EPOCHS
from worldmodels.models.vae import ConvVAE

# ── Reduced batch size to save memory ─────────────────────────────────────
BATCH_SIZE = 64

def get_rollout_files(rollout_dir):
    files = sorted(glob.glob(os.path.join(rollout_dir, "*.npz")))
    assert len(files) > 0, f"No rollout files in {rollout_dir}"
    return files

def frame_generator(files):
    """
    Yields one frame at a time from rollout files.
    Never loads more than one file into memory at once.
    """
    for f in files:
        data = np.load(f)
        for frame in data["obs"]:
            yield frame.astype(np.float32)

def make_dataset(files, batch_size):
    """
    tf.data pipeline that reads files lazily —
    only one file in memory at a time.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: frame_generator(files),
        output_signature=tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32)
    )
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_step(model, optimizer, x):
    with tf.GradientTape() as tape:
        total, recon, kl = model.compute_loss(x)
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total, recon, kl

def main():
    files    = get_rollout_files(ROLLOUT_DIR)
    print(f"Found {len(files)} rollout files. Training lazily (low memory)...")

    dataset   = make_dataset(files, BATCH_SIZE)
    model     = ConvVAE()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    log_path = os.path.join(VAE_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,step,total,recon,kl\n")

    step = 0
    for epoch in range(VAE_EPOCHS):
        losses = []
        for batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{VAE_EPOCHS}"):
            total, recon, kl = train_step(model, optimizer, batch)
            losses.append(total.numpy())
            if step % 200 == 0:
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{step},{total:.5f},{recon:.5f},{kl:.5f}\n")
            step += 1
        print(f"  Epoch {epoch+1}: mean_loss = {np.mean(losses):.5f}")
        model.save_weights_to()

    print("VAE training complete.")

if __name__ == "__main__":
    main()