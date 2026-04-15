import argparse
import cv2
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image
import gymnasium as gym
from worldmodels.models.mdn_rnn    import MDNRNN
from worldmodels.models.controller import Controller
from worldmodels.config import (LSTM_UNITS, MAX_EPISODE_STEPS,
                                 DINOV2_WEIGHTS, DINOV2_IMG_SIZE)


def load_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        img_size=DINOV2_IMG_SIZE
    )
    try:
        from timm.models import load_checkpoint
    except ImportError:
        from timm.models.helpers import load_checkpoint
    load_checkpoint(model, DINOV2_WEIGHTS, strict=False)
    model.eval().to(device)
    print(f"  DINOv2 loaded at {DINOV2_IMG_SIZE}x{DINOV2_IMG_SIZE} on {device}")
    return model, device


def run_episode(enc_model, device, transform, rnn, ctrl, seed=0):
    env    = gym.make("CarRacing-v2", continuous=True)
    obs, _ = env.reset(seed=seed)
    h      = np.zeros(LSTM_UNITS, dtype=np.float32)
    state  = rnn.get_initial_state(batch_size=1)
    total  = 0.0
    
    for _ in range(MAX_EPISODE_STEPS):
        frame = cv2.resize(obs, (DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                           interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        img   = Image.fromarray((frame * 255).astype(np.uint8))
        x     = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feats = enc_model.forward_features(x)
            z_full = feats[:, 0, :].cpu().numpy()[0]
            
            # --- 1. BOTTLENECK: Use only first 32 dims to match trained Controller ---
            z_ctrl = z_full[:32]
            
            # --- 2. SCALING: Standardize to match the training pipeline ---
            z_ctrl = (z_ctrl - np.mean(z_ctrl)) / (np.std(z_ctrl) + 1e-6)
            
            # --- 3. RNN INPUT: Use the full vector scaled by 2.0 (as trained) ---
            z_rnn = z_full * 2.0

        # Action logic now receives 32-dim z + 256-dim h (Total 288)
        action = ctrl.action(z_ctrl, h)
        
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        
        # Update RNN state with the full vector it was trained on
        h, state = rnn.step(z_rnn, action, state)
        
        if terminated or truncated:
            break
            
    env.close()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("Loading models...")
    enc_model, device = load_encoder()
    transform = T.Compose([
        T.Resize((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    rnn  = MDNRNN.load_from()
    
    # --- 4. DIMENSION SYNC: Initialize Controller with z_dim=32 ---
    ctrl = Controller(z_dim=32)
    ctrl.load() 

    rewards = []
    for ep in range(args.episodes):
        r = run_episode(enc_model, device, transform, rnn, ctrl, seed=ep)
        rewards.append(r)
        print(f"  Episode {ep+1:3d}: {r:7.2f}")

    print(f"\n{'='*40}")
    print(f"Mean +- Std : {np.mean(rewards):.1f} +- {np.std(rewards):.1f}")
    print(f"Best episode: {np.max(rewards):.2f}")
    print(f"Solved(>=900): {'YES' if np.mean(rewards) >= 900 else 'NO'}")


if __name__ == "__main__":
    main()