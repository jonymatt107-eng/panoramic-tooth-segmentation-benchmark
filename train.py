"""Patient-grouped development CV; untouched test set and FP32 training."""
from dataclasses import asdict
from pathlib import Path
import gc
import sys
import time
import importlib.metadata
import json
import platform
import random
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config, MODEL_NAMES, training_config
from dataset import BiomedicalDataset, make_splits, read_manifest, sha256, verify_locked_data
from losses_and_metrics import CombinedLoss, binary_metrics, soft_dice_loss, METRICS
from models import create_model, model_statistics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def environment(cfg: Config) -> dict:
    packages = {p: importlib.metadata.version(p) for p in
                ("torch", "torchvision", "segmentation-models-pytorch", "albumentations", "numpy", "pandas", "opencv-python", "scikit-learn")}
    hardware = platform.machine()
    if platform.system() == "Darwin":
        raw = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"], capture_output=True, text=True, check=True).stdout
        obj = json.loads(raw)["SPHardwareDataType"][0]
        hardware = {k: v for k, v in obj.items() if k in {"chip_type", "machine_model", "physical_memory", "number_processors"}}
    return {"packages": packages, "python": platform.python_version(), "platform": platform.platform(),
            "hardware": hardware, "device": str(cfg.device), "precision": "float32",
            "deterministic_algorithms": True, "threads": torch.get_num_threads()}


def run_epoch(model: torch.nn.Module, loader: DataLoader, cfg: Config,
              optimizer: torch.optim.Optimizer | None = None) -> dict:
    training = optimizer is not None
    model.train(training)
    criterion = CombinedLoss()
    loss_sum, dice_loss_sum, count, rows = 0.0, 0.0, 0, []
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False, disable=not sys.stdout.isatty(), desc="train" if training else "validation"):
            x, y = batch["image"].to(cfg.device), batch["mask"].to(cfg.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            valid = batch["valid"].to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if training:
                loss.backward()
                optimizer.step()
            n = len(x)
            loss_sum += loss.item() * n
            dice_loss_sum += soft_dice_loss(logits.detach(), y, valid).item() * n
            count += n
            metrics = binary_metrics(logits, y, cfg.threshold, valid)
            rows.extend([{**m, "patient_id": pid} for m, pid in zip(metrics, batch["patient_id"])])
    grouped = pd.DataFrame(rows).groupby("patient_id")[list(METRICS)].mean()
    metric_frame = pd.DataFrame(rows)[list(METRICS)]
    return {"loss": loss_sum / count, "dice_loss": dice_loss_sum / count,
            **grouped.mean().to_dict(),
            **{f"{k}_valid_images": int(v) for k, v in metric_frame.count().items()}}


def train(cfg: Config, resume: bool = False) -> None:
    seed_everything(cfg.seed)
    root = Path(cfg.output_dir).resolve()
    existing = root.exists()
    if existing and not resume:
        raise FileExistsError(f"Run directory exists: {root}; choose a new output_dir")
    if existing and Config.load(root / "config.json") != cfg:
        raise ValueError("Resume configuration differs from saved configuration")
    frame = make_splits(read_manifest(cfg.manifest), cfg)
    root.mkdir(parents=True, exist_ok=resume)
    if existing:
        locked = pd.read_csv(root / "splits.csv", dtype={"image_id": str, "patient_id": str})
        verify_locked_data(cfg, locked)
        if frame[["image_id", "split", "fold"]].to_dict("records") != locked[["image_id", "split", "fold"]].to_dict("records"):
            raise ValueError("Resume split mismatch")
    current_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    if existing:
        if json.loads((root / "source_hashes.json").read_text()) != current_hashes:
            raise ValueError("Source code changed since training began; preserve this run and choose a new one")
    else:
        cfg.save(root / "config.json")
        frame.to_csv(root / "splits.csv", index=False)
        (root / "environment.json").write_text(json.dumps(environment(cfg), indent=2))
        frame.groupby(["split", "fold"]).agg(images=("image_id", "size"), patients=("patient_id", "nunique"),
                   positive_images=("foreground", "sum")).to_csv(root / "cohort_counts.csv")
        (root / "source_hashes.json").write_text(json.dumps(current_hashes, indent=2))
    for name in MODEL_NAMES:
        model_dir = root / name
        model_dir.mkdir(exist_ok=resume)
        for fold in range(cfg.folds):
            complete = model_dir / f"fold_{fold}_complete.json"
            if resume and complete.exists():
                print(f"Resume: skip complete {name} fold {fold}", flush=True)
                continue
            seed_everything(cfg.seed + fold)
            dev = frame[frame.split == "dev"]
            tr, va = dev[dev.fold != fold], dev[dev.fold == fold]
            generator = torch.Generator().manual_seed(cfg.seed + fold)
            train_loader = DataLoader(BiomedicalDataset(tr, cfg, True), batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=0, generator=generator)
            val_loader = DataLoader(BiomedicalDataset(va, cfg), batch_size=cfg.batch_size, num_workers=0)
            model = create_model(name, cfg.encoder_weights).to(cfg.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
            best, stale, history = -float("inf"), 0, []
            for epoch in range(cfg.epochs):
                epoch_start = time.monotonic()
                train_metrics = run_epoch(model, train_loader, cfg, optimizer)
                val_metrics = run_epoch(model, val_loader, cfg)
                history.append({"epoch": epoch+1, "epoch_seconds": time.monotonic()-epoch_start, "lr": optimizer.param_groups[0]["lr"],
                                **{f"train_{k}": v for k, v in train_metrics.items()},
                                **{f"val_{k}": v for k, v in val_metrics.items()}})
                pd.DataFrame(history).to_csv(model_dir / f"history_fold_{fold}.csv", index=False)
                scheduler.step()
                if val_metrics["dice"] > best:
                    best, stale = val_metrics["dice"], 0
                    torch.save({"model_state": model.state_dict(), "model_name": name, "fold": fold,
                                "epoch": epoch+1, "val_dice": best, "selection": "maximum validation patient macro hard Dice", "config": asdict(cfg),
                                "statistics": model_statistics(model)}, model_dir / f"best_model_fold_{fold}.pth")
                else:
                    stale += 1
                print(f"{name} fold={fold} epoch={epoch+1} val_dice={val_metrics['dice']:.5f} val_dice_loss={val_metrics['dice_loss']:.5f} seconds={time.monotonic()-epoch_start:.1f}", flush=True)
                if stale >= cfg.patience:
                    break
            complete.write_text(json.dumps({"best_val_dice": best, "epochs": len(history)}))
            del optimizer, scheduler, model, train_loader, val_loader
            gc.collect()
            if cfg.device.type == "mps":
                torch.mps.empty_cache()
    (root / "TRAINING_COMPLETE").write_text("All architectures and folds completed.\n")

if __name__ == "__main__":
    train(training_config())
