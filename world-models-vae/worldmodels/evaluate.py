import argparse, cv2, numpy as np
import gymnasium as gym
from worldmodels.models.vae        import ConvVAE
from worldmodels.models.mdn_rnn    import MDNRNN
from worldmodels.models.controller import Controller
from worldmodels.config            import LSTM_UNITS, MAX_EPISODE_STEPS


def run_episode(vae, rnn, ctrl, seed=0):
    env   = gym.make("CarRacing-v2", continuous=True)
    obs, _ = env.reset(seed=seed)
    h         = np.zeros(LSTM_UNITS, dtype=np.float32)
    rnn_state = rnn.get_initial_state(batch_size=1)
    total_r   = 0.0
    for _ in range(MAX_EPISODE_STEPS):
        frame  = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
        frame  = frame.astype(np.float32) / 255.0
        z      = vae.encode_frame(frame)
        action = ctrl.action(z, h)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r
        h, rnn_state = rnn.step(z, action, rnn_state)
        if terminated or truncated:
            break
    env.close()
    return total_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("Loading models...")
    vae  = ConvVAE.load_from()
    rnn  = MDNRNN.load_from()
    ctrl = Controller.load()

    rewards = []
    for ep in range(args.episodes):
        r = run_episode(vae, rnn, ctrl, seed=ep)
        rewards.append(r)
        print(f"  Episode {ep+1:3d}: {r:7.2f}")

    print(f"\n{'='*40}")
    print(f"Mean ± Std : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Solved(≥900): {'YES ✓' if np.mean(rewards) >= 900 else 'NO ✗'}")

if __name__ == "__main__":
    main()