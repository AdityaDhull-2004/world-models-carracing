import os, glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from worldmodels.config import (LATENT_DIR, MEMORY_DIR,
                                 RNN_BATCH, RNN_EPOCHS, SEQ_LEN)
from worldmodels.models.mdn_rnn import MDNRNN


def sequence_generator(latent_dir, seq_len):
    """
    Yields (input, target) sequences one at a time.
    Never loads more than one file into memory at once.
    """
    files = sorted(glob.glob(os.path.join(latent_dir, "*.npz")))
    assert len(files) > 0, f"No latent files in {latent_dir}"

    for f in files:
        data    = np.load(f)
        mus     = data["mu"].astype(np.float32)
        logvars = data["logvar"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        T       = len(mus)

        if T < seq_len + 1:
            continue

        # Sample z ~ N(mu, sigma)
        eps = np.random.randn(*mus.shape).astype(np.float32)
        zs  = mus + np.exp(0.5 * logvars) * eps

        for start in range(0, T - seq_len, 50):
            end = start + seq_len
            inp = np.concatenate([zs[start:end],
                                  actions[start:end]], axis=-1)
            tgt = zs[start+1:end+1]
            yield inp, tgt


def make_dataset(latent_dir, seq_len, batch_size):
    """Lazy tf.data pipeline — reads one file at a time."""
    # Determine shapes from config
    from worldmodels.config import Z_DIM, ACTION_DIM
    inp_shape = (seq_len, Z_DIM + ACTION_DIM)
    tgt_shape = (seq_len, Z_DIM)

    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(latent_dir, seq_len),
        output_signature=(
            tf.TensorSpec(shape=inp_shape, dtype=tf.float32),
            tf.TensorSpec(shape=tgt_shape, dtype=tf.float32),
        )
    )
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_step(model, optimizer, inp, tgt):
    with tf.GradientTape() as tape:
        pi, mu, log_sigma, _ = model(inp)
        loss = model.mdn_loss(pi, mu, log_sigma, tgt)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def main():
    print("Building lazy dataset...")
    dataset   = make_dataset(LATENT_DIR, SEQ_LEN, RNN_BATCH)
    model     = MDNRNN()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    log_path = os.path.join(MEMORY_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,step,loss\n")

    step = 0
    for epoch in range(RNN_EPOCHS):
        losses = []
        for inp_b, tgt_b in tqdm(dataset,
                                  desc=f"Epoch {epoch+1}/{RNN_EPOCHS}"):
            loss = train_step(model, optimizer, inp_b, tgt_b)
            losses.append(loss.numpy())
            if step % 200 == 0:
                with open(log_path, "a") as f:
                    f.write(f"{epoch},{step},{loss:.5f}\n")
            step += 1

        mean_loss = np.mean(losses)
        print(f"  Epoch {epoch+1}: mean_loss = {mean_loss:.5f}")
        model.save_weights_to()

    print("MDN-RNN training complete.")


if __name__ == "__main__":
    main()