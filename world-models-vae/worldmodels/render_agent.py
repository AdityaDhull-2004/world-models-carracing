"""
Record the trained agent driving as a video file.
"""
import cv2
import numpy as np
import gymnasium as gym
from worldmodels.models.vae        import ConvVAE
from worldmodels.models.mdn_rnn    import MDNRNN
from worldmodels.models.controller import Controller
from worldmodels.config            import LSTM_UNITS, MAX_EPISODE_STEPS


def record_episode(vae, rnn, ctrl, seed=0, output_path="agent_driving.mp4"):
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    h         = np.zeros(LSTM_UNITS, dtype=np.float32)
    rnn_state = rnn.get_initial_state(batch_size=1)
    total_r   = 0.0
    frames    = []

    for step in range(MAX_EPISODE_STEPS):
        # Record raw frame
        raw_frame = env.render()
        frames.append(raw_frame)

        # Encode and act
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
    print(f"Episode reward: {total_r:.2f}")
    print(f"Total frames: {len(frames)}")

    # Save as video
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved → {output_path}")
    return total_r


def main():
    print("Loading models...")
    vae  = ConvVAE.load_from()
    rnn  = MDNRNN.load_from()
    ctrl = Controller.load()

    # Record 3 episodes
    for ep in range(74, 75):
        print(f"\nRecording episode {ep+1}...")
        record_episode(vae, rnn, ctrl,
                      seed=ep,
                      output_path=f"episode_{ep+1}.mp4")

if __name__ == "__main__":
    main()