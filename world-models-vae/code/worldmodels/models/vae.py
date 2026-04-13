"""
ConvVAE — Vision (V) model.
Encoder : 64x64x3 → Conv×4 → Dense → mu, logvar  (z_dim each)
Decoder : z        → Dense(1024) → ConvTranspose×4 → 64x64x3
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from worldmodels.config import Z_DIM, IMAGE_SIZE, VAE_DIR


class ConvVAE(keras.Model):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        # Encoder
        self.enc_conv1 = keras.layers.Conv2D(32,  4, strides=2, activation="relu", padding="valid")
        self.enc_conv2 = keras.layers.Conv2D(64,  4, strides=2, activation="relu", padding="valid")
        self.enc_conv3 = keras.layers.Conv2D(128, 4, strides=2, activation="relu", padding="valid")
        self.enc_conv4 = keras.layers.Conv2D(256, 4, strides=2, activation="relu", padding="valid")
        self.flatten   = keras.layers.Flatten()
        self.fc_mu     = keras.layers.Dense(z_dim)
        self.fc_logvar = keras.layers.Dense(z_dim)
        # Decoder
        self.fc_decode = keras.layers.Dense(1024)
        self.dec_conv1 = keras.layers.Conv2DTranspose(128, 5, strides=2, activation="relu",    padding="valid")
        self.dec_conv2 = keras.layers.Conv2DTranspose(64,  5, strides=2, activation="relu",    padding="valid")
        self.dec_conv3 = keras.layers.Conv2DTranspose(32,  6, strides=2, activation="relu",    padding="valid")
        self.dec_conv4 = keras.layers.Conv2DTranspose(3,   6, strides=2, activation="sigmoid", padding="valid")

    def encode(self, x):
        h = self.enc_conv4(self.enc_conv3(self.enc_conv2(self.enc_conv1(x))))
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mu))

    def decode(self, z):
        h = tf.reshape(self.fc_decode(z), (-1, 1, 1, 1024))
        h = self.dec_conv4(self.dec_conv3(self.dec_conv2(self.dec_conv1(h))))
        return h[:, :IMAGE_SIZE, :IMAGE_SIZE, :]

    def call(self, x, training=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if training else mu
        return self.decode(z), mu, logvar

    def compute_loss(self, x, kl_weight=0.01):
        recon, mu, logvar = self(x, training=True)
        recon_loss = tf.reduce_mean(tf.square(x - recon))
        logvar = tf.clip_by_value(logvar, -4.0, 4.0)
        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + logvar - tf.square(mu) - tf.exp(logvar)
        )
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

    def encode_frame(self, frame):
        """(64,64,3) numpy → (z_dim,) numpy"""
        mu, _ = self.encode(tf.expand_dims(tf.constant(frame), 0))
        return mu.numpy()[0]

    def _build(self):
        self(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)), training=False)

    def save_weights_to(self, path=None):
        path = path or os.path.join(VAE_DIR, "vae_weights")
        self.save_weights(path)
        print(f"  VAE saved → {path}")

    @classmethod
    def load_from(cls, path=None, z_dim=Z_DIM):
        path  = path or os.path.join(VAE_DIR, "vae_weights")
        model = cls(z_dim=z_dim)
        model._build()
        model.load_weights(path)
        print(f"  VAE loaded ← {path}")
        return model