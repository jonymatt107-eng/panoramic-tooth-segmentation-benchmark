"""Central experiment configuration; FP32 is the portable MPS default."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import torch

MODEL_NAMES = ("unet_resnet34", "unet_efficientnet_b0", "segformer_b0")

@dataclass(frozen=True)
class Config:
    require_mps: bool = False
    manifest: str = "data/manifest.csv"
    output_dir: str = "runs/main"
    image_size: int = 512
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    folds: int = 5
    patience: int = 7
    epochs: int = 35
    test_fraction: float = 0.2
    num_workers: int = 0
    threshold: float = 0.5
    encoder_weights: str | None = "imagenet"
    noise_repeats: int = 3
    bootstrap_samples: int = 2000

    def __post_init__(self) -> None:
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if min(self.batch_size, self.epochs, self.patience, self.noise_repeats, self.bootstrap_samples) < 1:
            raise ValueError("Batch, epochs, patience, repeats and bootstrap count must be positive")
        if self.folds < 2 or not 0 < self.test_fraction < 1 or not 0 < self.threshold < 1:
            raise ValueError("Invalid folds, test fraction or threshold")
        if self.num_workers != 0:
            raise ValueError("Use num_workers=0 for reproducible single-process augmentation RNG")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")

    @property
    def device(self) -> torch.device:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if self.require_mps and device.type != "mps":
            raise RuntimeError("This experiment requires MPS; CPU fallback is prohibited")
        return device

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls(**json.loads(path.read_text()))


def training_config() -> Config:
    parser = argparse.ArgumentParser(description="Train binary segmentation architectures")
    parser.add_argument("--config", type=Path, help="JSON containing Config fields")
    args = parser.parse_args()
    return Config.load(args.config) if args.config else Config()
