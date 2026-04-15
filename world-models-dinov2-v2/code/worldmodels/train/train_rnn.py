import os, glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from worldmodels.config import (LATENT_DIR, MEMORY_DIR,
                                 RNN_BATCH, RNN_EPOCHS, SEQ_LEN)
from worldmodels.models.mdn_rnn import MDNRNN


def sequence_generator(latent_dir, seq_len):
    files = sorted(glob.glob(os.path.join(latent_dir, "*.npz")))
    for f in files:
        data = np.load(f)
        # Assuming 'mu' contains your DINOv2 features
        zs = data["mu"].astype(np.float32) 
        actions = data["actions"].astype(np.float32)
        
        # Apply scaling here if you haven't in the encoder
        # zs = zs * 2.0 

        T = len(zs)
        if T < seq_len + 1: continue
        
        for start in range(0, T - seq_len, 50):
            end = start + seq_len
            inp = np.concatenate([zs[start:end], actions[start:end]], axis=-1)
            tgt = zs[start+1:end+1]
            yield inp, tgt


def make_dataset(latent_dir, seq_len, batch_size):
    from worldmodels.config import Z_DIM, ACTION_DIM
    dataset = tf.data.Dataset.from_generator(
        lambda: sequence_generator(latent_dir, seq_len),
        output_signature=(
            tf.TensorSpec(shape=(seq_len, Z_DIM + ACTION_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(seq_len, Z_DIM), dtype=tf.float32),
        )
    )
    return (dataset.shuffle(2000)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))


def train_step(model, optimizer, inp, tgt):
    with tf.GradientTape() as tape:
        pi, mu, log_sigma, _ = model(inp)
        loss = model.mdn_loss(pi, mu, log_sigma, tgt)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def main():
    print("Building dataset...")
    dataset = make_dataset(LATENT_DIR, SEQ_LEN, RNN_BATCH)
    model   = MDNRNN()

    # HIGH learning rate (1e-3 not 1e-4) — key fix for flat loss problem.
    # With DINOv2 384-dim features, 1e-4 is too small to make meaningful
    # progress in only 10 epochs. 1e-3 gives 10x larger gradient steps.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    log_path = os.path.join(MEMORY_DIR, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,step,loss\n")

    best_loss = float('inf')
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
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save_weights_to()
            print(f"  New best: {best_loss:.5f}")

    model.save_weights_to()
    print(f"MDN-RNN training complete. Best loss: {best_loss:.5f}")


if __name__ == "__main__":
    main()
