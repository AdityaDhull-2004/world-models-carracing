"""
Collect random rollouts from CarRacing-v2.
Saves each episode as a .npz file:
  obs     : (T, 64, 64, 3) float32
  actions : (T, 3)         float32
"""
import os, argparse, numpy as np, multiprocessing as mp
from tqdm import tqdm

def resize_frame(frame):
    import cv2
    return cv2.resize(frame, (64, 64),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

def run_episode(args):
    ep_id, max_steps, seed, rollout_dir = args
    import gymnasium as gym
    env = gym.make("CarRacing-v2", continuous=True)
    obs, _ = env.reset(seed=seed)
    observations, actions = [], []
    for _ in range(max_steps):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        observations.append(resize_frame(obs))
        actions.append(action.astype(np.float32))
        obs = next_obs
        if terminated or truncated:
            break
    env.close()
    path = os.path.join(rollout_dir, f"rollout_{ep_id:05d}.npz")
    np.savez_compressed(path,
                        obs=np.stack(observations),
                        actions=np.stack(actions))
    return ep_id

def main():
    from worldmodels.config import ROLLOUT_DIR, MAX_EPISODE_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=10000)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=MAX_EPISODE_STEPS)
    args = parser.parse_args()

    tasks = [(i, args.max_steps, i, ROLLOUT_DIR) for i in range(args.episodes)]
    with mp.Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(run_episode, tasks),
                      total=args.episodes, desc="Collecting rollouts"):
            pass
    print(f"Done. Rollouts saved to: {ROLLOUT_DIR}")

if __name__ == "__main__":
    main()