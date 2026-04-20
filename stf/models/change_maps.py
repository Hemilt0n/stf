from __future__ import annotations

import torch
import torch.nn.functional as F


def build_soft_change_map(
    coarse_img_01,
    coarse_img_02,
    target_spatial_shape,
    smooth_kernel_size: int = 3,
    power: float = 1.0,
):
    change_map = torch.mean(torch.abs(coarse_img_02 - coarse_img_01), dim=1, keepdim=True)
    if change_map.shape[-2:] != target_spatial_shape:
        change_map = F.interpolate(
            change_map,
            size=target_spatial_shape,
            mode="bilinear",
            align_corners=False,
        )
    if smooth_kernel_size > 1:
        padding = smooth_kernel_size // 2
        change_map = F.avg_pool2d(
            change_map,
            kernel_size=smooth_kernel_size,
            stride=1,
            padding=padding,
        )
    denom = change_map.detach().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    norm_change = change_map / denom
    soft_change = norm_change / (1.0 + norm_change)
    if power != 1.0:
        soft_change = soft_change.clamp(min=0.0).pow(power)
    return soft_change.clamp(0.0, 1.0)


def summarize_trust_by_change(trust_map: torch.Tensor, change_map: torch.Tensor) -> dict[str, float]:
    if change_map.shape[-2:] != trust_map.shape[-2:]:
        change_map = F.interpolate(
            change_map,
            size=trust_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if change_map.shape[1] != 1:
        change_map = torch.mean(change_map, dim=1, keepdim=True)

    trust = trust_map.detach()
    changed = change_map.detach().clamp(0.0, 1.0)
    unchanged = (1.0 - changed).clamp(0.0, 1.0)

    def _weighted_mean(weight: torch.Tensor) -> float:
        denom = weight.sum().clamp(min=1e-6)
        return float(((trust * weight).sum() / denom).item())

    return {
        "trust_mean": float(trust.mean().item()),
        "trust_changed": _weighted_mean(changed),
        "trust_unchanged": _weighted_mean(unchanged),
    }
