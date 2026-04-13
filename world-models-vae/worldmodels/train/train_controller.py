import os, numpy as np, multiprocessing as mp
from worldmodels.config import (CONTROLLER_DIR, POP_SIZE, ROLLOUTS_PER_AGENT,
                                 MAX_GENERATIONS, MAX_EPISODE_STEPS, LSTM_UNITS)
from worldmodels.models.controller import Controller


def evaluate_params(args):
    """Runs in a worker process. Loads models fresh (TF is not fork-safe)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for workers
    import cv2, numpy as np, tensorflow as tf
    import gymnasium as gym
    from worldmodels.models.vae        import ConvVAE
    from worldmodels.models.mdn_rnn    import MDNRNN
    from worldmodels.models.controller import Controller
    from worldmodels.config            import LSTM_UNITS, MAX_EPISODE_STEPS

    params_vec, num_rollouts, generation = args
    vae  = ConvVAE.load_from()
    rnn  = MDNRNN.load_from()
    ctrl = Controller()
    ctrl.set_params(params_vec)

    rewards = []
    for i in range(num_rollouts):
        env   = gym.make("CarRacing-v2", continuous=True)
        obs, _ = env.reset(seed=generation * 1000 + i)
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
        rewards.append(total_r)

    return -np.mean(rewards)   # CMA-ES minimises


def main():
    import cma
    ctrl     = Controller()
    n_params = ctrl.num_params
    print(f"Controller parameters: {n_params}")

    es = cma.CMAEvolutionStrategy(
        n_params * [0], 0.1,
        {"popsize": POP_SIZE, "maxiter": MAX_GENERATIONS, "verbose": -9}
    )

    log_path    = os.path.join(CONTROLLER_DIR, "rewards.log")
    best_reward = -np.inf
    generation  = 0

    with open(log_path, "w") as f:
        f.write("generation,mean_reward,best_reward,sigma\n")

    while not es.stop():
        solutions = es.ask()
        tasks = [(s, ROLLOUTS_PER_AGENT, generation) for s in solutions]
        n_workers = min(POP_SIZE, mp.cpu_count())
        with mp.Pool(n_workers) as pool:
            neg_rewards = pool.map(evaluate_params, tasks)
        print(f"  Using {n_workers} workers for {POP_SIZE} agents")
        es.tell(solutions, neg_rewards)

        rewards = [-r for r in neg_rewards]
        mean_r  = np.mean(rewards)
        max_r   = np.max(rewards)
        if max_r > best_reward:
            best_reward = max_r
            ctrl.set_params(solutions[np.argmax(rewards)])
            ctrl.save()

        print(f"Gen {generation:4d} | mean={mean_r:7.2f} | "
              f"best={best_reward:7.2f} | σ={es.sigma:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{generation},{mean_r:.2f},{best_reward:.2f},{es.sigma:.4f}\n")
        generation += 1

    print(f"Done. Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()