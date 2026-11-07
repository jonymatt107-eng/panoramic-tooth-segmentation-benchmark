"""Stable loss, per-image confusion metrics and synchronized forward timing."""
import time
import numpy as np
import torch
from torch import nn

METRICS = ("dice", "iou", "sensitivity", "specificity", "precision")


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(target)
    p, y = (logits.float().sigmoid()*valid).flatten(1), (target.float()*valid).flatten(1)
    return (1 - (2 * (p*y).sum(1) + 1e-6) / (p.sum(1) + y.sum(1) + 1e-6)).mean()


class CombinedLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        elementwise = nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
        bce = ((elementwise*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
        return .5*bce + .5*soft_dice_loss(logits, target, valid)


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                   valid: torch.Tensor | None = None) -> list[dict]:
    p, y = (logits.detach().sigmoid() >= threshold).flatten(1), (target > 0.5).flatten(1)
    v = torch.ones_like(y) if valid is None else (valid > .5).flatten(1)
    tp = (p & y & v).sum(1).cpu().numpy().astype(float)
    tn = (~p & ~y & v).sum(1).cpu().numpy().astype(float)
    fp = (p & ~y & v).sum(1).cpu().numpy().astype(float)
    fn = (~p & y & v).sum(1).cpu().numpy().astype(float)
    def ratio(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")
    return [dict(dice=ratio(2*a, 2*a+c+d) if 2*a+c+d else 1.0,
                 iou=ratio(a, a+c+d) if a+c+d else 1.0,
                 sensitivity=ratio(a, a+d), specificity=ratio(b, b+c), precision=ratio(a, a+c),
                 tp=int(a), tn=int(b), fp=int(c), fn=int(d), foreground=int(a+d > 0))
            for a, b, c, d in zip(tp, tn, fp, fn)]


def benchmark_latency(model: nn.Module, device: torch.device, image_size: int = 512,
                      warmup: int = 100, iterations: int = 200) -> dict:
    if warmup < 0 or iterations < 1:
        raise ValueError("Invalid latency iteration counts")
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(iterations):
            sync()
            start = time.perf_counter_ns()
            model(x)
            sync()
            times.append((time.perf_counter_ns() - start) / 1e6)
    mean = float(np.mean(times))
    return {"latency_ms": mean, "latency_sd_ms": float(np.std(times)),
            "latency_p50_ms": float(np.median(times)), "latency_p95_ms": float(np.percentile(times, 95)),
            "fps": 1000 / mean, "latency_batch_size": 1, "warmup": warmup, "iterations": iterations}
