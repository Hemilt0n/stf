from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage import io as skio


def _denormalize(img: np.ndarray, normalize_scale, normalize_mode):
    if normalize_mode == 1:
        return img * normalize_scale
    if normalize_mode == 2:
        return (img + 1.0) / 2.0 * normalize_scale
    return img


def save_prediction_image(
    tensor,
    output_dir: Path,
    filename: str,
    normalize_scale,
    normalize_mode,
) -> None:
    arr = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    arr = _denormalize(arr, normalize_scale, normalize_mode)
    arr = np.clip(arr, 0, normalize_scale).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_dir / filename, arr)


def save_show_image(
    show_tensors,
    pred_tensor,
    output_dir: Path,
    filename: str,
    normalize_mode,
    show_bands=(2, 1, 0),
    img_interval: int = 10,
) -> None:
    show_len = len(show_tensors)
    h_num, w_num = 3, max(show_len // 2, 1)
    _, c, h, w = pred_tensor.shape
    canvas = np.zeros(
        (
            (h + img_interval) * h_num + img_interval,
            (w + img_interval) * w_num + img_interval,
            3,
        ),
        dtype=np.uint8,
    )

    for h_index in range(h_num - 1):
        for w_index in range(w_num):
            idx = h_index * w_num + w_index
            if idx >= len(show_tensors):
                continue
            sub = show_tensors[idx][0].detach().cpu().numpy().transpose(1, 2, 0)
            if normalize_mode == 1:
                sub = sub * 255.0
            elif normalize_mode == 2:
                sub = (sub + 1.0) / 2.0 * 255.0
            if c > 3:
                sub = sub[:, :, show_bands]
            sub = np.clip(sub, 0, 255).astype(np.uint8)
            sub = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
            y0 = img_interval * (h_index + 1) + h_index * h
            x0 = img_interval * (w_index + 1) + w_index * w
            canvas[y0 : y0 + h, x0 : x0 + w, :] = sub

    pred = pred_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    if normalize_mode == 1:
        pred = pred * 255.0
    elif normalize_mode == 2:
        pred = (pred + 1.0) / 2.0 * 255.0
    if c > 3:
        pred = pred[:, :, show_bands]
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    y0 = img_interval * h_num + h * (h_num - 1)
    x0 = img_interval * w_num + w * (w_num - 1)
    canvas[y0 : y0 + h, x0 : x0 + w, :] = pred

    output_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(output_dir / filename, canvas)


def save_trust_map_image(
    trust_tensor,
    output_dir: Path,
    filename: str,
    change_tensor=None,
) -> None:
    trust = trust_tensor[0].detach().cpu().numpy()
    if trust.ndim == 3:
        trust = trust[0]
    trust_img = np.clip(trust, 0.0, 1.0)
    trust_img = (trust_img * 255.0).astype(np.uint8)

    canvas = trust_img
    if change_tensor is not None:
        change = change_tensor[0].detach().cpu().numpy()
        if change.ndim == 3:
            change = change[0]
        change_img = np.clip(change, 0.0, 1.0)
        change_img = (change_img * 255.0).astype(np.uint8)
        canvas = np.concatenate((change_img, trust_img), axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    skio.imsave(output_dir / filename, canvas)
