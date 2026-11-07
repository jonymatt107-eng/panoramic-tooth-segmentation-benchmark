"""Segmentation Grad-CAM on the deepest spatial encoder output, including MiT."""
from pathlib import Path
import argparse
import gc
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config, MODEL_NAMES
from dataset import read_pair, letterbox_pair
from models import load_checkpoint
from robustness_eval import read_locked, tensor_image
from train import seed_everything


class GradCAM:
    """Capture bottleneck tensors through the encoder output instead of brittle layer names."""
    def __init__(self, model: torch.nn.Module, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold
        self.activation: torch.Tensor | None = None
        self.handle = model.encoder.register_forward_hook(self.capture)

    def capture(self, module: torch.nn.Module, inputs: tuple, output: list[torch.Tensor]) -> None:
        self.activation = output[-1]

    def close(self) -> None:
        self.handle.remove()

    def __call__(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x.requires_grad_(True))
            if valid is None:
                valid = torch.ones_like(logits)
            region = (logits.detach().sigmoid() >= self.threshold).float()*valid
            if not region.any():
                region = valid
            score = (logits * region).sum() / region.sum()
            if self.activation is None or self.activation.ndim != 4:
                raise ValueError("Expected BCHW encoder activation")
            gradient = torch.autograd.grad(score, self.activation)[0]
            cam = (gradient.mean((2, 3), keepdim=True) * self.activation).sum(1, keepdim=True).relu()
            cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = (cam*valid)[0, 0].detach().cpu().numpy()
            cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-8)
            probability = logits[0, 0].detach().sigmoid().cpu().numpy()
        self.activation = None
        return probability, cam


def explain(root: Path, split: str, fold: int, count: int) -> None:
    cfg = Config.load(root / "config.json")
    seed_everything(cfg.seed)
    if not 0 <= fold < cfg.folds or count < 1:
        raise ValueError("Invalid fold or count")
    frame = read_locked(root, cfg)
    frame = frame[frame.split == split].head(count)
    if frame.empty:
        raise ValueError("No images in requested partition")
    out = root / f"explainability_{split}_fold_{fold}"
    out.mkdir(exist_ok=True)
    # Load one architecture at a time to conserve unified memory.
    images, masks, validity = [], [], []
    for row in frame.itertuples():
        image, mask = read_pair(row.image_path, row.mask_path)
        image, mask, valid = letterbox_pair(image, mask, cfg)
        images.append(image.astype(np.float32)/255)
        masks.append(mask)
        validity.append(valid)
    results = {}
    for name in MODEL_NAMES:
        model = load_checkpoint(str(root / name / f"best_model_fold_{fold}.pth"), name, cfg.device)
        cam = GradCAM(model, cfg.threshold)
        try:
            results[name] = [cam(tensor_image(image).to(cfg.device), torch.from_numpy(valid).unsqueeze(0).unsqueeze(0).to(cfg.device))
                             for image, valid in zip(images, validity)]
        finally:
            cam.close()
        del cam, model
        gc.collect()
        if cfg.device.type == "mps":
            torch.mps.empty_cache()
    for i, row in enumerate(frame.itertuples()):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(images[i]); axes[0].set_title("Original image")
        axes[1].imshow(masks[i], cmap="gray", vmin=0, vmax=1); axes[1].set_title("Reference mask")
        for ax, name in zip(axes[2:], MODEL_NAMES):
            probability, heat = results[name][i]
            ax.imshow(images[i]); ax.imshow(heat, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            prediction = (probability >= cfg.threshold) & (validity[i] > .5)
            if prediction.any() and not prediction.all():
                ax.contour(prediction, levels=[0.5], colors="cyan", linewidths=0.7)
            ax.set_title(name.replace("_", " "), fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Image {i+1}; fold {fold}; cyan: prediction boundary; heat: positive-logit attribution")
        fig.tight_layout()
        fig.savefig(out / f"panel_{i+1:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    frame[["image_id", "patient_id"]].to_csv(out / "panel_index.csv", index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("runs/main"))
    p.add_argument("--split", choices=("test", "external"), default="test")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()
    explain(args.run, args.split, args.fold, args.count)
