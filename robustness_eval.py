"""Paired, reproducible post-resize corruption evaluation of every fold model."""
from pathlib import Path
import argparse
import csv
import gc
import sys
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from config import Config, MODEL_NAMES
from dataset import MEAN, STD, read_pair, verify_locked_data, letterbox_pair
from losses_and_metrics import binary_metrics, benchmark_latency
from models import load_checkpoint, model_statistics
from train import seed_everything, environment


def conditions() -> list[tuple[str, int, float]]:
    return [("clean", 0, 0.0)] + [(kind, level, value) for kind, values in
        {"gaussian_noise": [0.01, 0.05, 0.1], "gaussian_blur": [3, 5, 7],
         "motion_blur": [3, 5, 7], "underexposure": [1.25, 1.75, 2.5],
         "resolution": [0.75, 0.5, 0.25]}.items() for level, value in enumerate(values, 1)]


def corrupt(image: np.ndarray, kind: str, value: float, rng: np.random.Generator,
            level: int = 0) -> np.ndarray:
    """RGB float32 [0,1] input; corrupt intensities only, preserving mask geometry."""
    x = image.copy()
    if kind == "gaussian_noise":
        x += rng.normal(0, value, x.shape).astype(np.float32)
    elif kind == "gaussian_blur":
        x = cv2.GaussianBlur(x, (int(value), int(value)), sigmaX=0)
    elif kind == "motion_blur":
        kernel = np.zeros((int(value), int(value)), np.float32)
        kernel[int(value)//2, :] = 1 / int(value)
        x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    elif kind == "underexposure":
        # Joint gamma and multiplicative contrast/dynamic-range attenuation.
        x = np.power(x, value) * {1: 0.9, 2: 0.7, 3: 0.5}[level]
    elif kind == "resolution":
        h, w = x.shape[:2]
        small = cv2.resize(x, (max(1, round(w*value)), max(1, round(h*value))), interpolation=cv2.INTER_LINEAR)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind != "clean":
        raise ValueError(f"Unknown corruption: {kind}")
    return np.clip(x, 0, 1).astype(np.float32)


def corruption_rng(seed: int, image_id: str, kind: str, repeat: int) -> np.random.Generator:
    # Omit architecture, fold and severity: paired realizations, scaled across severity.
    token = hashlib.sha256(f"{seed}|{image_id}|{kind}|{repeat}".encode()).digest()
    return np.random.default_rng(int.from_bytes(token[:8], "little"))


def tensor_image(image: np.ndarray) -> torch.Tensor:
    x = (image - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)


def read_locked(root: Path, cfg: Config) -> pd.DataFrame:
    frame = pd.read_csv(root / "splits.csv", dtype={"image_id": str, "patient_id": str}, keep_default_na=False)
    verify_locked_data(cfg, frame)
    return frame


def evaluate(root: Path, split: str = "test", warmup: int = 100, iterations: int = 200) -> None:
    cfg = Config.load(root / "config.json")
    seed_everything(cfg.seed)
    if not (root / "TRAINING_COMPLETE").exists():
        raise ValueError("Complete all training folds before evaluation")
    frame = read_locked(root, cfg)
    frame = frame[frame.split == split]
    if frame.empty:
        raise ValueError(f"No {split} patients in locked manifest")
    out = root / f"evaluation_{split}"
    out.mkdir()  # Refuse to overwrite a previous evaluation, including an interrupted one.
    (out / "environment.json").write_text(json.dumps(environment(cfg), indent=2))
    timing = []
    fields = ["model", "fold", "split", "image_id", "patient_id", "corruption", "severity", "value", "repeat",
              "dice", "iou", "sensitivity", "specificity", "precision", "tp", "tn", "fp", "fn", "foreground"]
    with (out / "per_image.partial.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in MODEL_NAMES:
            for fold in range(cfg.folds):
                path = root / name / f"best_model_fold_{fold}.pth"
                model = load_checkpoint(str(path), name, cfg.device)
                timing.append({"model": name, "fold": fold, **model_statistics(model),
                               **benchmark_latency(model, cfg.device, cfg.image_size, warmup, iterations)})
                with torch.inference_mode():
                    for row in tqdm(frame.itertuples(), total=len(frame), disable=not sys.stdout.isatty(), desc=f"{name}/{fold}/{split}"):
                        image, mask = read_pair(row.image_path, row.mask_path)
                        image, mask, valid = letterbox_pair(image, mask, cfg)
                        image = image.astype(np.float32)/255
                        ys, xs = np.where(valid > .5)
                        y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
                        valid_tensor = torch.from_numpy(valid).unsqueeze(0).unsqueeze(0).to(cfg.device)
                        target = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(cfg.device)
                        for kind, level, value in conditions():
                            repeats = cfg.noise_repeats if kind == "gaussian_noise" else 1
                            for repeat in range(repeats):
                                x = image.copy()
                                x[y0:y1, x0:x1] = corrupt(image[y0:y1, x0:x1], kind, value,
                                    corruption_rng(cfg.seed, row.image_id, kind, repeat), level)
                                logits = model(tensor_image(x).to(cfg.device))
                                writer.writerow({"model": name, "fold": fold, "split": split,
                                    "image_id": row.image_id, "patient_id": row.patient_id,
                                    "corruption": kind, "severity": level, "value": value, "repeat": repeat,
                                    **binary_metrics(logits, target, cfg.threshold, valid_tensor)[0]})
                        stream.flush()
                del model
                gc.collect()
                if cfg.device.type == "mps":
                    torch.mps.empty_cache()
    pd.DataFrame(timing).to_csv(out / "efficiency.csv", index=False)
    (out / "per_image.partial.csv").rename(out / "per_image.csv")
    (out / "EVALUATION_COMPLETE").write_text("Evaluation completed.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--split", choices=("test", "external"), default="test")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.run, args.split, args.warmup, args.iterations)

if __name__ == "__main__":
    main()
