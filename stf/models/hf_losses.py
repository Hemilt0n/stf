import torch
import torch.nn.functional as F


def compute_change_map(coarse_img_01, coarse_img_02, target_spatial_shape):
    change_map = torch.mean(torch.abs(coarse_img_02 - coarse_img_01), dim=1, keepdim=True)
    if change_map.shape[-2:] != target_spatial_shape:
        change_map = F.interpolate(
            change_map,
            size=target_spatial_shape,
            mode="bilinear",
            align_corners=False,
        )
    return change_map


def build_change_mask(
    coarse_img_01,
    coarse_img_02,
    target_spatial_shape,
    strategy="quantile",
    quantile=0.8,
    threshold=0.0,
    topk_ratio=0.2,
):
    change_map = compute_change_map(coarse_img_01, coarse_img_02, target_spatial_shape)
    strategy = strategy.lower()

    if strategy in ("quantile", "percentile"):
        q = float(max(0.0, min(0.999, quantile)))
        flat = change_map.flatten(1)
        tau = torch.quantile(flat, q=q, dim=1, keepdim=True).view(-1, 1, 1, 1)
        return (change_map > tau).type_as(change_map)

    if strategy in ("fixed", "threshold"):
        return (change_map > float(threshold)).type_as(change_map)

    if strategy in ("topk",):
        ratio = float(max(0.0, min(1.0, topk_ratio)))
        if ratio <= 0.0:
            return torch.zeros_like(change_map)
        if ratio >= 1.0:
            return torch.ones_like(change_map)

        flat = change_map.flatten(1)
        num_pixels = flat.shape[1]
        k = max(1, int(round(ratio * num_pixels)))
        tau = torch.topk(flat, k=k, dim=1).values[:, -1:].view(-1, 1, 1, 1)
        return (change_map >= tau).type_as(change_map)

    raise ValueError(f"invalid change mask strategy: {strategy}")


def masked_mean(value_map, mask=None):
    if mask is None:
        return value_map.mean()

    if mask.shape[1] == 1 and value_map.shape[1] != 1:
        mask = mask.expand(-1, value_map.shape[1], -1, -1)
    mask = mask.type_as(value_map)

    weighted = value_map * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom


def gradient_l1_loss(pred, target, mask=None):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    mask_x = None
    mask_y = None
    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = masked_mean(torch.abs(pred_dx - target_dx), mask_x)
    loss_y = masked_mean(torch.abs(pred_dy - target_dy), mask_y)
    return 0.5 * (loss_x + loss_y)


def _laplacian_response(x):
    channels = x.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=channels)


def laplacian_pyramid_l1_loss(pred, target, mask=None, num_scales=3):
    total = pred.new_tensor(0.0)
    pred_curr = pred
    target_curr = target
    mask_curr = mask
    scales = 0

    for _ in range(max(1, int(num_scales))):
        total = total + masked_mean(
            torch.abs(_laplacian_response(pred_curr) - _laplacian_response(target_curr)),
            mask_curr,
        )
        scales += 1

        if min(pred_curr.shape[-2], pred_curr.shape[-1]) < 4:
            break

        pred_curr = F.avg_pool2d(pred_curr, kernel_size=2, stride=2)
        target_curr = F.avg_pool2d(target_curr, kernel_size=2, stride=2)
        if mask_curr is not None:
            mask_curr = F.max_pool2d(mask_curr, kernel_size=2, stride=2)

    return total / float(scales)


def ranking_prefer_target_loss(pred, target, reference, mask=None, margin=0.0):
    d_target = torch.mean(torch.abs(pred - target), dim=1, keepdim=True)
    d_reference = torch.mean(torch.abs(pred - reference), dim=1, keepdim=True)
    ranking = F.relu(d_target - d_reference + float(margin))
    return masked_mean(ranking, mask)
