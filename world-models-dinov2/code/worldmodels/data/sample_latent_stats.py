import os, glob
import numpy as np
from tqdm import tqdm
from worldmodels.config import ROLLOUT_DIR, LATENT_DIR


def main():
    import torch
    from worldmodels.models.dinov2_encoder import load_dinov2_encoder
    encoder = load_dinov2_encoder()

    # fp16 for faster encoding — halves memory and time
    encoder.model = encoder.model.half()
    print("  Using fp16 for faster GPU encoding")

    files = sorted(glob.glob(os.path.join(ROLLOUT_DIR, "*.npz")))
    assert len(files) > 0, f"No rollouts found in {ROLLOUT_DIR}"

    done  = set(os.listdir(LATENT_DIR))
    files = [f for f in files if os.path.basename(f) not in done]
    print(f"Encoding {len(files)} remaining rollouts at 70x70 (fp16)...")

    batch_size = 256
    for fpath in tqdm(files):
        data    = np.load(fpath)
        obs     = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)

        zs = []
        for i in range(0, len(obs), batch_size):
            batch = obs[i:i+batch_size]
            import torchvision.transforms as T
            from PIL import Image
            imgs = torch.stack([
                encoder.transform(Image.fromarray(f.astype(np.uint8)))
                for f in batch
            ]).half().to(encoder.device)
            with torch.no_grad():
                feats = encoder.model.forward_features(imgs)
                z = feats[:, 0, :].float().cpu().numpy()
            zs.append(z)
            torch.cuda.empty_cache()

        zs = np.concatenate(zs, axis=0).astype(np.float32)
        out = os.path.join(LATENT_DIR, os.path.basename(fpath))
        np.savez_compressed(
            out,
            mu=zs,
            logvar=np.zeros_like(zs),
            actions=actions
        )

    print(f"Done. {len(os.listdir(LATENT_DIR))} files saved to {LATENT_DIR}")


if __name__ == "__main__":
    main()
