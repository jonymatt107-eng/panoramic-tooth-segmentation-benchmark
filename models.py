"""Matching three-channel input, one-channel binary-logit model interfaces."""
import torch
from torch import nn
import segmentation_models_pytorch as smp
from config import MODEL_NAMES
from mps_compat import enable_deterministic_segformer


def create_model(name: str, encoder_weights: str | None = "imagenet") -> nn.Module:
    common = dict(encoder_weights=encoder_weights, in_channels=3, classes=1, activation=None)
    if name == "unet_resnet34":
        return smp.Unet(encoder_name="resnet34", **common)
    if name == "unet_efficientnet_b0":
        return smp.Unet(encoder_name="efficientnet-b0", **common)
    if name == "segformer_b0":
        return enable_deterministic_segformer(smp.Segformer(encoder_name="mit_b0", **common))
    raise ValueError(f"Unknown model {name}; choose {MODEL_NAMES}")


def model_statistics(model: nn.Module) -> dict[str, int | float]:
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
            "state_mb": sum(p.numel() * p.element_size() for p in model.state_dict().values()) / 1e6}


def load_checkpoint(path: str, name: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != name:
        raise ValueError("Checkpoint architecture mismatch")
    model = create_model(name, encoder_weights=None)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()
