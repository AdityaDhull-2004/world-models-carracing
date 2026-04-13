"""MDN-RNN — Memory (M) model. LSTM + Mixture Density Network."""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from worldmodels.config import Z_DIM, ACTION_DIM, LSTM_UNITS, NUM_MIXTURES, MEMORY_DIR

LOG_2PI = np.log(2 * np.pi).astype(np.float32)


class MDNRNN(keras.Model):
    def __init__(self, z_dim=Z_DIM, action_dim=ACTION_DIM,
                 lstm_units=LSTM_UNITS, num_mixtures=NUM_MIXTURES):
        super().__init__()
        self.z_dim        = z_dim
        self.action_dim   = action_dim
        self.lstm_units   = lstm_units
        self.num_mixtures = num_mixtures
        self.lstm   = keras.layers.LSTM(lstm_units, return_sequences=True,
                                        return_state=True)
        self.mdn_fc = keras.layers.Dense(num_mixtures * (1 + z_dim * 2))

    def call(self, inputs, initial_state=None):
        lstm_out, h, c = self.lstm(inputs, initial_state=initial_state)
        raw = self.mdn_fc(lstm_out)
        K, Z = self.num_mixtures, self.z_dim
        T    = tf.shape(raw)[1]
        pi_logits = raw[:, :, :K]
        mu        = tf.reshape(raw[:, :, K:K+K*Z],       (-1, T, K, Z))
        log_sigma = tf.reshape(raw[:, :, K+K*Z:K+2*K*Z], (-1, T, K, Z))
        return pi_logits, mu, log_sigma, (h, c)

    def mdn_loss(self, pi_logits, mu, log_sigma, z_next):
        Z         = self.z_dim
        z_next    = tf.expand_dims(z_next, 2)
        log_sigma = tf.clip_by_value(log_sigma, -4.0, 4.0)
        sigma     = tf.exp(log_sigma)
        log_prob  = -0.5 * (
            tf.reduce_sum(tf.square((z_next - mu) / (sigma + 1e-8)), axis=-1)
            + tf.cast(Z, tf.float32) * LOG_2PI
            + 2.0 * tf.reduce_sum(log_sigma, axis=-1)
        )
        log_pi      = tf.nn.log_softmax(pi_logits, axis=-1)
        log_mixture = tf.reduce_logsumexp(log_pi + log_prob, axis=-1)
        return -tf.reduce_mean(log_mixture)

    def step(self, z, action, state):
        """Single inference step → (h_vector, new_state)"""
        inp = np.concatenate([z, action])[None, None, :].astype(np.float32)
        _, _, _, new_state = self(tf.constant(inp), initial_state=state)
        return new_state[0].numpy()[0], new_state

    def get_initial_state(self, batch_size=1):
        return (tf.zeros((batch_size, self.lstm_units)),
                tf.zeros((batch_size, self.lstm_units)))

    def _build(self):
        self(tf.zeros((1, 1, self.z_dim + self.action_dim)))

    def save_weights_to(self, path=None):
        path = path or os.path.join(MEMORY_DIR, "mdn_rnn_weights")
        self.save_weights(path)
        print(f"  MDN-RNN saved → {path}")

    @classmethod
    def load_from(cls, path=None, **kwargs):
        path  = path or os.path.join(MEMORY_DIR, "mdn_rnn_weights")
        model = cls(**kwargs)
        model._build()
        model.load_weights(path)
        print(f"  MDN-RNN loaded ← {path}")
        return model