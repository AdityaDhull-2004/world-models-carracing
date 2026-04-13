"""Controller (C) — linear policy, 867 parameters, trained with CMA-ES."""
import os
import numpy as np
from worldmodels.config import Z_DIM, ACTION_DIM, LSTM_UNITS, CONTROLLER_DIR


class Controller:
    def __init__(self, z_dim=Z_DIM, h_dim=LSTM_UNITS, action_dim=ACTION_DIM):
        self.z_dim      = z_dim
        self.h_dim      = h_dim
        self.action_dim = action_dim
        self.input_dim  = z_dim + h_dim
        self.W = np.zeros((action_dim, self.input_dim), dtype=np.float32)
        self.b = np.zeros(action_dim, dtype=np.float32)

    @property
    def num_params(self):
        return self.action_dim * self.input_dim + self.action_dim

    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b])

    def set_params(self, params):
        n      = self.action_dim * self.input_dim
        self.W = params[:n].reshape(self.action_dim, self.input_dim)
        self.b = params[n:n + self.action_dim]

    def action(self, z, h):
        raw   = np.tanh(self.W @ np.concatenate([z, h]) + self.b)
        steer = raw[0]
        gas   = (raw[1] + 1.0) / 2.0
        brake = (raw[2] + 1.0) / 2.0
        return np.array([steer, gas, brake], dtype=np.float32)

    def save(self, path=None):
        path = path or os.path.join(CONTROLLER_DIR, "controller.npz")
        np.savez(path, W=self.W, b=self.b)
        print(f"  Controller saved → {path}")

    @classmethod
    def load(cls, path=None, **kwargs):
        path = path or os.path.join(CONTROLLER_DIR, "controller.npz")
        data = np.load(path)
        c    = cls(**kwargs)
        c.W  = data["W"]
        c.b  = data["b"]
        print(f"  Controller loaded ← {path}")
        return c