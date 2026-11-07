"""SMP 0.5.0 MiT attention with equivalent unbind-based key/value extraction.

The computation follows segmentation_models_pytorch.encoders.mix_transformer.Attention
(MIT license, copyright segmentation-models-pytorch contributors). The only
mathematical change is none: kv.unbind(0) replaces kv[0], kv[1]. This avoids an
MPS index-accumulation backward kernel lacking a deterministic implementation.
Parameter names, shapes and checkpoint compatibility remain unchanged.
"""
from types import MethodType
import torch
from torch import nn
from segmentation_models_pytorch.encoders.mix_transformer import Attention


def attention_forward(self: Attention, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, n, channels = x.shape
    q = self.q(x).reshape(batch_size, n, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
    reduced = x
    if self.sr_ratio > 1:
        reduced = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        reduced = self.sr(reduced).reshape(batch_size, channels, -1).permute(0, 2, 1)
        reduced = self.norm(reduced)
    kv = self.kv(reduced).reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(dim=0)
    attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
    attention = self.attn_drop(attention)
    x = (attention @ v).transpose(1, 2).reshape(batch_size, n, channels)
    return self.proj_drop(self.proj(x))


def enable_deterministic_mit(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, Attention):
            module.forward = MethodType(attention_forward, module)
    return model


class DeterministicBilinear(torch.autograd.Function):
    """Native interpolation forward, separable matrix transpose gradient.

    Bilinear interpolation is linear in input pixels. Its adjoint is
    W_height.T @ gradient @ W_width. Matrix products avoid atomic scatter
    accumulation in the MPS interpolation backward. No tensors move to CPU.
    """
    @staticmethod
    def forward(ctx: object, x: torch.Tensor, height: int, width: int, align: bool) -> torch.Tensor:
        ctx.input_shape = x.shape[-2:]
        ctx.align = align
        return torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=align)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple:
        h, w = ctx.input_shape
        oh, ow = gradient.shape[-2:]
        def weights(n_in: int, n_out: int) -> torch.Tensor:
            index = torch.arange(n_out, device=gradient.device, dtype=gradient.dtype)
            if ctx.align:
                position = index * ((n_in-1) / (n_out-1)) if n_out > 1 else index * 0
            else:
                position = ((index+.5) * n_in/n_out - .5).clamp(0, n_in-1)
            grid = torch.arange(n_in, device=gradient.device, dtype=gradient.dtype)
            return (1 - (position[:, None] - grid[None, :]).abs()).clamp(min=0)
        wh, ww = weights(h, oh), weights(w, ow)
        result = torch.matmul(wh.transpose(0, 1), torch.matmul(gradient, ww))
        return result, None, None, None


def resize_bilinear(x: torch.Tensor, size: tuple[int, int], align: bool = False) -> torch.Tensor:
    if x.device.type == "mps" and x.requires_grad and torch.is_grad_enabled():
        return DeterministicBilinear.apply(x, size[0], size[1], align)
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=align)


class BilinearHead(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize_bilinear(x, (x.shape[-2]*self.scale, x.shape[-1]*self.scale), align=True)


def decoder_forward(self: nn.Module, features: list[torch.Tensor]) -> torch.Tensor:
    target = tuple(dim//4 for dim in features[0].shape[2:])
    stages = features[2:] if features[1].size(1) == 0 else features[1:]
    stages = stages[::-1]
    resized = [resize_bilinear(layer(feature), target) for layer, feature in zip(self.mlp_stage, stages)]
    return self.fuse_stage(torch.cat(resized, dim=1))


def enable_deterministic_segformer(model: nn.Module) -> nn.Module:
    enable_deterministic_mit(model)
    model.decoder.forward = MethodType(decoder_forward, model.decoder)
    model.segmentation_head[1] = BilinearHead(scale=4)
    return model
