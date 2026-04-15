import os
import numpy as np
import multiprocessing as mp
from worldmodels.config import (CONTROLLER_DIR, POP_SIZE, ROLLOUTS_PER_AGENT,
                                 MAX_GENERATIONS, MAX_EPISODE_STEPS, LSTM_UNITS)
from worldmodels.models.controller import Controller


def evaluate_params(args):
    """
    Worker: timm loads DINOv2 on GPU (PyTorch).
    TF RNN runs on CPU.
    Includes 2.0x scaling for RNN and 32-D bottleneck for Controller.
    Includes Reward Shaping to accelerate Mean Reward growth.
    """
    import os
    import cv2
    import numpy as np
    import torch
    import timm
    import torchvision.transforms as T
    from PIL import Image
    import gymnasium as gym
    import tensorflow as tf
    
    # --- 1. THE FIX: Hide GPU from TensorFlow, but NOT from PyTorch ---
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass

    from worldmodels.models.mdn_rnn import MDNRNN
    from worldmodels.models.controller import Controller
    from worldmodels.config import (
        LSTM_UNITS, MAX_EPISODE_STEPS,
        DINOV2_WEIGHTS, DINOV2_IMG_SIZE
    )

    params_vec, num_rollouts, generation = args
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. Load Models Once Per Worker ---
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

    # Load RNN weights (Running on CPU)
    rnn_w = MDNRNN.load_from()
    
    # Initialize Controller with 32-dim bottleneck to match z_ctrl
    ctrl_w = Controller(z_dim=32)
    ctrl_w.set_params(params_vec)

    # --- 3. Rollout Loop ---
    rewards = []
    for i in range(num_rollouts):
        env = gym.make("CarRacing-v2", continuous=True)
        obs, _ = env.reset(seed=generation * 1000 + i)
        
        h = np.zeros(LSTM_UNITS, dtype=np.float32)
        state = rnn_w.get_initial_state(batch_size=1)
        total_reward = 0.0

        for _ in range(MAX_EPISODE_STEPS):
            # Pre-process frame for DINOv2
            frame = cv2.resize(obs, (DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                               interpolation=cv2.INTER_AREA)
            frame = frame.astype(np.float32) / 255.0
            img = Image.fromarray((frame * 255).astype(np.uint8))
            x = transform(img).unsqueeze(0).to(device)
            
            # Feature Extraction (GPU)
            with torch.no_grad():
                feats = enc_model.forward_features(x)
                z_full = feats[:, 0, :].cpu().numpy()[0]
                
                # Apply the scaling the RNN expects (2.0x)
                z_rnn = z_full * 2.0 
                
                # Create the focused 32-D version for the Controller
                z_ctrl = z_full[:32]
                z_ctrl = (z_ctrl - np.mean(z_ctrl)) / (np.std(z_ctrl) + 1e-6)

            # Control Action
            action = ctrl_w.action(z_ctrl, h)
            
            # Environment Step
            obs, r, terminated, truncated, _ = env.step(action)
            
            # --- REWARD SHAPING ADDED HERE ---
            # Amplify positive progress (hitting tiles)
            if r > 0:
                r *= 1.5 
            # Penalize lack of movement or driving off-track slightly more
            else:
                r -= 0.1
            
            total_reward += r
            
            # RNN State Update (Full features)
            h, state = rnn_w.step(z_rnn, action, state)
            
            if terminated or truncated:
                break

        env.close()
        rewards.append(total_reward)

    # Return negative mean because CMA-ES minimizes the objective
    return -np.mean(rewards)


def main():
    import cma
    import multiprocessing as mp

    # Ensure the controller used for counting params is set to 32
    ctrl = Controller(z_dim=32)
    n_params = ctrl.num_params
    
    # --- Performance Configuration ---
    # With 24 cores and an RTX 3050, 4 workers is a safe balance for VRAM.
    # If you see VRAM is still low in nvidia-smi, you can try 6 or 8.
    num_workers = 8
    
    print(f"Controller parameters: {n_params}")
    print(f"Population: {POP_SIZE} | Rollouts/agent: {ROLLOUTS_PER_AGENT}")
    print(f"Parallel Workers: {num_workers}")
    print(f"Evals per gen: {POP_SIZE * ROLLOUTS_PER_AGENT}")
    print(f"Initial sigma: 0.05")

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(
        n_params * [0], 0.05,
        {"popsize": POP_SIZE, "maxiter": MAX_GENERATIONS, "verbose": -9}
    )

    log_path    = os.path.join(CONTROLLER_DIR, "rewards.log")
    best_reward = -np.inf
    generation  = 0

    # Write header
    with open(log_path, "w") as f:
        f.write("generation,mean_reward,best_reward,sigma\n")

    try:
        while not es.stop():
            solutions = es.ask()
            tasks     = [(s, ROLLOUTS_PER_AGENT, generation) for s in solutions]

            # --- Faster Parallel Execution ---
            with mp.Pool(num_workers) as pool:
                neg_rewards = pool.map(evaluate_params, tasks)

            es.tell(solutions, neg_rewards)
            
            # Process results
            rewards = [-r for r in neg_rewards]
            mean_r  = np.mean(rewards)
            max_r   = np.max(rewards)

            # Checkpoint the best model
            if max_r > best_reward:
                best_reward = max_r
                best_idx    = np.argmax(rewards)
                ctrl.set_params(solutions[best_idx])
                ctrl.save()
                print(f" ---> New Best! Reward: {best_reward:.2f} (Saved weights)")

            # Log progress
            status_line = (f"Gen {generation:4d} | mean={mean_r:7.2f} | "
                           f"best={best_reward:7.2f} | sigma={es.sigma:.4f}")
            print(status_line)

            with open(log_path, "a") as f:
                f.write(f"{generation},{mean_r:.2f},"
                        f"{best_reward:.2f},{es.sigma:.4f}\n")
            
            generation += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    finally:
        print(f"Done. Final Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
