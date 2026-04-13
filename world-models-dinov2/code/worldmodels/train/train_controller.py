import os
import numpy as np
import multiprocessing as mp
from worldmodels.config import (CONTROLLER_DIR, POP_SIZE, ROLLOUTS_PER_AGENT,
                                 MAX_GENERATIONS, MAX_EPISODE_STEPS, LSTM_UNITS)
from worldmodels.models.controller import Controller


def evaluate_params(args):
    """
    Worker: DINOv2 on GPU (PyTorch), RNN on CPU (TF).
    Uses timm + load_checkpoint — no pickling errors, no dim mismatch.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TF on CPU only

    import cv2
    import numpy as np
    import torch
    import timm
    import torchvision.transforms as T
    from PIL import Image
    import gymnasium as gym
    from worldmodels.models.mdn_rnn    import MDNRNN
    from worldmodels.models.controller import Controller
    from worldmodels.config import (
        LSTM_UNITS, MAX_EPISODE_STEPS,
        DINOV2_WEIGHTS, DINOV2_IMG_SIZE
    )

    params_vec, num_rollouts, generation = args

    # Load DINOv2 via timm on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_model = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        img_size=DINOV2_IMG_SIZE
    )
    try:
        from timm.models import load_checkpoint
    except ImportError:
        from timm.models.helpers import load_checkpoint
    load_checkpoint(enc_model, DINOV2_WEIGHTS, strict=False)
    enc_model.eval().to(device)

    transform = T.Compose([
        T.Resize((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    rnn_w  = MDNRNN.load_from()
    ctrl_w = Controller()
    ctrl_w.set_params(params_vec)

    rewards = []
    for i in range(num_rollouts):
        env    = gym.make("CarRacing-v2", continuous=True)
        obs, _ = env.reset(seed=generation * 1000 + i)
        h      = np.zeros(LSTM_UNITS, dtype=np.float32)
        state  = rnn_w.get_initial_state(batch_size=1)
        total  = 0.0

        for _ in range(MAX_EPISODE_STEPS):
            frame = cv2.resize(obs, (DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                               interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32) / 255.0
            img   = Image.fromarray((frame * 255).astype(np.uint8))
            x     = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = enc_model.forward_features(x)
                z = feats[:, 0, :].cpu().numpy()[0]

            action = ctrl_w.action(z, h)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            h, state = rnn_w.step(z, action, state)
            if terminated or truncated:
                break

        env.close()
        rewards.append(total)

    return -np.mean(rewards)


def main():
    import cma

    ctrl     = Controller()
    n_params = ctrl.num_params
    print(f"Controller parameters: {n_params}")
    print(f"Population: {POP_SIZE}  Rollouts/agent: {ROLLOUTS_PER_AGENT}")
    print(f"Evals per generation: {POP_SIZE * ROLLOUTS_PER_AGENT}")
    print(f"Max generations: {MAX_GENERATIONS}")

    es = cma.CMAEvolutionStrategy(
        n_params * [0], 0.3,
        {"popsize": POP_SIZE, "maxiter": MAX_GENERATIONS, "verbose": -9}
    )

    log_path    = os.path.join(CONTROLLER_DIR, "rewards.log")
    best_reward = -np.inf
    generation  = 0

    with open(log_path, "w") as f:
        f.write("generation,mean_reward,best_reward,sigma\n")

    while not es.stop():
        solutions = es.ask()
        tasks     = [(s, ROLLOUTS_PER_AGENT, generation) for s in solutions]

        with mp.Pool(1) as pool:
            neg_rewards = pool.map(evaluate_params, tasks)

        es.tell(solutions, neg_rewards)
        rewards = [-r for r in neg_rewards]
        mean_r  = np.mean(rewards)
        max_r   = np.max(rewards)

        if max_r > best_reward:
            best_reward = max_r
            ctrl.set_params(solutions[np.argmax(rewards)])
            ctrl.save()

        print(f"Gen {generation:4d} | mean={mean_r:7.2f} | "
              f"best={best_reward:7.2f} | sigma={es.sigma:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{generation},{mean_r:.2f},"
                    f"{best_reward:.2f},{es.sigma:.4f}\n")
        generation += 1

    print(f"Done. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
