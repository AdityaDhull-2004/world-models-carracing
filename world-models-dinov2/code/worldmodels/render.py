"""
Render DINOv2 World Models agent playing CarRacing-v2.
Saves episode frames as an MP4.
Usage: python -m worldmodels.render --episodes 3 --output renders/
"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import imageio

from worldmodels.models.mdn_rnn import MDNRNN
from worldmodels.models.controller import Controller
from worldmodels.models.dinov2_encoder import load_dinov2_encoder
from worldmodels.config import LSTM_UNITS, MAX_EPISODE_STEPS

def run_and_record(encoder, rnn, ctrl, seed=0):
    """Run one episode and return frames + total reward."""
    # Use rgb_array so we can capture the frames for the video
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    
    h = np.zeros(LSTM_UNITS, dtype=np.float32)
    state = rnn.get_initial_state(batch_size=1)
    total = 0.0
    frames = []

    for step in range(MAX_EPISODE_STEPS):
        # 1. Capture the high-quality frame for the video
        render_frame = env.render()
        frames.append(render_frame)

        # 2. Encode the observation using the 70x70 helper
        # Normalize obs to [0, 1] as expected by our encoder helper
        z = encoder.encode_frame(obs / 255.0)

        # 3. Get action from Controller
        action = ctrl.action(z, h)
        
        # 4. Step environment
        obs, r, terminated, truncated, _ = env.step(action)
        total += r
        
        # 5. Update RNN state
        h, state = rnn.step(z, action, state)
        
        if terminated or truncated:
            break

    env.close()
    return frames, total

def main():
    parser = argparse.ArgumentParser(description="Render DINOv2 World Models agent")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render")
    parser.add_argument("--output", type=str, default="renders", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--gif", action="store_true", help="Also save as GIF")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading models...")
    # Using the helper ensures the 70x70 interpolation logic is applied
    encoder = load_dinov2_encoder()
    rnn = MDNRNN.load_from()
    ctrl = Controller.load()

    rewards = []
    for ep in range(args.episodes):
        print(f"\nRendering episode {ep+1}/{args.episodes}...")
        
        frames, reward = run_and_record(encoder, rnn, ctrl, seed=ep)
        rewards.append(reward)
        
        print(f"  Reward: {reward:.2f}  Frames: {len(frames)}")

        # Save as MP4
        mp4_path = os.path.join(args.output, f"episode_{ep+1}_r{int(reward)}.mp4")
        try:
            # quality=8 is a good balance for file size/clarity
            writer = imageio.get_writer(mp4_path, fps=args.fps, codec='libx264', quality=8)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"  Saved MP4: {mp4_path}")
        except Exception as e:
            print(f"  Failed to save MP4: {e}")

        # Optionally save as GIF
        if args.gif:
            gif_path = os.path.join(args.output, f"episode_{ep+1}_r{int(reward)}.gif")
            imageio.mimsave(gif_path, frames[::2], fps=args.fps//2)
            print(f"  Saved GIF: {gif_path}")

    print(f"\n{'='*40}")
    print(f"Results: Mean reward: {np.mean(rewards):.1f} over {args.episodes} episodes.")
    print(f"Files saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()