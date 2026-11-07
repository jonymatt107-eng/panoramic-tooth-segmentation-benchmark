"""Synthetic integration check; generated scores are never clinical evidence."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from config import Config
from dataset import read_manifest
from train import train
from robustness_eval import evaluate
from explainability import explain
from generate_paper_artifacts import generate


def smoke(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / "synthetic"
    data.mkdir()
    rng = np.random.default_rng(42)
    rows = []
    for i in range(14):
        mask = np.zeros((64, 64), np.uint8)
        if i % 3:
            cv2.circle(mask, (28+i%8, 32), 8+i%5, 255, -1)
        image = rng.integers(0, 100, (64, 64, 3), dtype=np.uint8)
        image[mask > 0] += 120
        cv2.imwrite(str(data / f"image_{i}.png"), image)
        cv2.imwrite(str(data / f"mask_{i}.png"), mask)
        rows.append(dict(image_id=f"image_{i:02d}", patient_id=f"patient_{i:02d}",
                         image_path=f"image_{i}.png", mask_path=f"mask_{i}.png",
                         split="dev" if i < 10 else "test" if i < 12 else "external"))
    manifest = data / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    read_manifest(str(manifest))
    cfg = Config(manifest=str(manifest), output_dir=str(root / "run"), image_size=64,
                 batch_size=2, folds=2, epochs=1, patience=1, encoder_weights=None,
                 noise_repeats=1, bootstrap_samples=20)
    train(cfg)
    evaluate(Path(cfg.output_dir), warmup=1, iterations=2)
    generate(Path(cfg.output_dir), "test")
    explain(Path(cfg.output_dir), "test", fold=0, count=1)
    print("SYNTHETIC INTEGRATION PASSED; not a scientific benchmark")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True, help="New scratch directory")
    smoke(p.parse_args().output)
