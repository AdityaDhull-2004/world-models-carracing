"""
DINOv2 ViT-Small/14 at 70x70 resolution.
70x70 -> 5x5=25 patches (14*5=70, exact multiple).
Uses timm load_checkpoint to handle positional embedding
interpolation automatically — no dimension mismatch.
Uses timm (NOT torch.hub) — no pickling errors.
"""
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from worldmodels.config import DINOV2_WEIGHTS, DINOV2_IMG_SIZE


class DINOv2Encoder:
    def __init__(self, weights_path=None, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  DINOv2 device: {self.device}")

        import timm
        self.model = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=DINOV2_IMG_SIZE
        )
        path = weights_path or DINOV2_WEIGHTS
        self._load_weights(path)
        self.model.eval().to(self.device)
        n_patches = (DINOV2_IMG_SIZE // 14) ** 2
        print(f"  DINOv2 loaded at {DINOV2_IMG_SIZE}x{DINOV2_IMG_SIZE}")
        print(f"  Patches: {DINOV2_IMG_SIZE//14}x{DINOV2_IMG_SIZE//14} = {n_patches}")

        self.transform = T.Compose([
            T.Resize((DINOV2_IMG_SIZE, DINOV2_IMG_SIZE),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found: {path}")
        try:
            from timm.models import load_checkpoint
        except ImportError:
            from timm.models.helpers import load_checkpoint
        load_checkpoint(self.model, path, strict=False)
        print(f"  Weights loaded successfully from {path}")

    def encode_frame(self, frame):
        """(H,W,3) float32 [0,1] -> (384,) float32"""
        img = Image.fromarray((frame * 255).astype(np.uint8))
        x   = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(x)
            z = feats[:, 0, :]
        return z.cpu().numpy()[0].astype(np.float32)

    def encode_batch(self, frames):
        """(B,H,W,3) float32 [0,1] -> (B,384) float32"""
        imgs = torch.stack([
            self.transform(Image.fromarray((f * 255).astype(np.uint8)))
            for f in frames
        ]).to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(imgs)
            zs = feats[:, 0, :]
        return zs.cpu().numpy().astype(np.float32)


def load_dinov2_encoder(weights_path=None):
    return DINOv2Encoder(weights_path=weights_path)
