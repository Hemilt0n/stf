from __future__ import annotations

import re
from pathlib import Path

import torch


_EPOCH_PATTERN = re.compile(r"model_epoch_(\d+)\.pth$")


def parse_epoch_from_path(path: str | Path) -> int | None:
    match = _EPOCH_PATTERN.search(str(path))
    return int(match.group(1)) if match else None


def load_legacy_checkpoint(
    checkpoint_path: str | Path,
    model,
    optimizer=None,
    ema=None,
    device: str = "cuda",
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" not in checkpoint:
        raise KeyError("Legacy checkpoint is missing 'model' key")

    model.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    epoch = checkpoint.get("epoch")
    if epoch is None:
        epoch = parse_epoch_from_path(checkpoint_path)

    return {
        "epoch": epoch,
        "has_optimizer": "optimizer" in checkpoint,
        "has_ema": "ema" in checkpoint,
        "raw": checkpoint,
    }
