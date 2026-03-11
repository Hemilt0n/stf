import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .hf_losses import (
    build_change_mask,
    compute_change_map,
    gradient_l1_loss,
    laplacian_pyramid_l1_loss,
    ranking_prefer_target_loss,
)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def compute_volume(x):
    """计算图像体积（可被重写以实现不同的体积定义）"""
    return x.sum(dim=(1, 2, 3))  # [b]


def maybe_apply_condition_dropout(condition, dropout_p: float, training: bool):
    if dropout_p <= 0.0 or not training:
        return condition
    keep_mask = (
        torch.rand(condition.shape[0], 1, 1, 1, device=condition.device) >= dropout_p
    ).type_as(condition)
    return condition * keep_mask


def build_change_weight_map(
    coarse_img_01, coarse_img_02, target_spatial_shape, change_loss_weight: float
):
    if change_loss_weight <= 0.0:
        return None
    change_map = compute_change_map(
        coarse_img_01, coarse_img_02, target_spatial_shape=target_spatial_shape
    )
    denom = change_map.detach().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    norm_change = change_map / denom
    return 1.0 + change_loss_weight * norm_change


def coarse_consistency_loss(
    pred_fine_img_02, coarse_img_02, loss_type: str = 'l1'
):
    pred_coarse = pred_fine_img_02
    if pred_coarse.shape[-2:] != coarse_img_02.shape[-2:]:
        pred_coarse = F.interpolate(
            pred_coarse,
            size=coarse_img_02.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    if loss_type == 'l2':
        return F.mse_loss(pred_coarse, coarse_img_02)
    return F.l1_loss(pred_coarse, coarse_img_02)


class FlowMatching(nn.Module):
    def __init__(
        self,
        model,
        loss_type='l2',
        num_steps=20,
        change_loss_weight=0.0,
        coarse_consistency_weight=0.0,
        coarse_consistency_loss_type='l1',
        grad_loss_weight=0.0,
        lap_loss_weight=0.0,
        lap_num_scales=3,
        ranking_loss_weight=0.0,
        ranking_margin=0.0,
        hf_mask_strategy='quantile',
        hf_mask_quantile=0.8,
        hf_mask_threshold=0.0,
        hf_mask_topk_ratio=0.2,
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.change_loss_weight = change_loss_weight
        self.coarse_consistency_weight = coarse_consistency_weight
        self.coarse_consistency_loss_type = coarse_consistency_loss_type
        self.grad_loss_weight = grad_loss_weight
        self.lap_loss_weight = lap_loss_weight
        self.lap_num_scales = lap_num_scales
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_margin = ranking_margin
        self.hf_mask_strategy = hf_mask_strategy
        self.hf_mask_quantile = hf_mask_quantile
        self.hf_mask_threshold = hf_mask_threshold
        self.hf_mask_topk_ratio = hf_mask_topk_ratio

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    @property
    def use_hf_losses(self):
        return (
            self.grad_loss_weight > 0.0
            or self.lap_loss_weight > 0.0
            or self.ranking_loss_weight > 0.0
        )

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02):
        b = fine_img_01.shape[0]
        device = fine_img_01.device

        # Sample time t
        t = torch.rand(b, device=device).type_as(fine_img_01)
        t = rearrange(t, 'b -> b 1 1 1')

        # Define the path x_t
        x_t = (1 - t) * fine_img_01 + t * fine_img_02

        # Define the target vector field u_t
        u_t = fine_img_02 - fine_img_01

        # Predict the vector field
        pred_u_t = self.model(
            coarse_img_01, coarse_img_02, fine_img_01, x_t, t.squeeze()
        )

        # Base reconstruction loss in vector-field space.
        per_pixel_loss = self.loss_fn(pred_u_t, u_t, reduction='none')
        weight_map = build_change_weight_map(
            coarse_img_01,
            coarse_img_02,
            target_spatial_shape=u_t.shape[-2:],
            change_loss_weight=self.change_loss_weight,
        )
        if weight_map is not None:
            per_pixel_loss = per_pixel_loss * weight_map

        loss = per_pixel_loss.mean()

        need_pred_fine = self.coarse_consistency_weight > 0.0 or self.use_hf_losses
        pred_fine_img_02 = fine_img_01 + pred_u_t if need_pred_fine else None

        if self.coarse_consistency_weight > 0.0:
            cst_loss = coarse_consistency_loss(
                pred_fine_img_02,
                coarse_img_02,
                loss_type=self.coarse_consistency_loss_type,
            )
            loss = loss + self.coarse_consistency_weight * cst_loss

        if self.use_hf_losses:
            change_mask = build_change_mask(
                coarse_img_01,
                coarse_img_02,
                target_spatial_shape=fine_img_02.shape[-2:],
                strategy=self.hf_mask_strategy,
                quantile=self.hf_mask_quantile,
                threshold=self.hf_mask_threshold,
                topk_ratio=self.hf_mask_topk_ratio,
            )

            if self.grad_loss_weight > 0.0:
                loss = loss + self.grad_loss_weight * gradient_l1_loss(
                    pred_fine_img_02, fine_img_02, mask=change_mask
                )

            if self.lap_loss_weight > 0.0:
                loss = loss + self.lap_loss_weight * laplacian_pyramid_l1_loss(
                    pred_fine_img_02,
                    fine_img_02,
                    mask=change_mask,
                    num_scales=self.lap_num_scales,
                )

            if self.ranking_loss_weight > 0.0:
                loss = loss + self.ranking_loss_weight * ranking_prefer_target_loss(
                    pred_fine_img_02,
                    fine_img_02,
                    fine_img_01,
                    mask=change_mask,
                    margin=self.ranking_margin,
                )
        return loss

    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        device = fine_img_01.device
        x_t = fine_img_01.clone()
        dt = 1.0 / self.num_steps

        for t_step in range(self.num_steps):
            t = torch.full((fine_img_01.shape[0],), t_step * dt, device=device).type_as(fine_img_01)
            pred_u_t = self.model(
                coarse_img_01, coarse_img_02, fine_img_01, x_t, t
            )
            x_t = x_t + pred_u_t * dt
        
        return x_t


class GaussianFlowMatching(nn.Module):
    """
    Flow matching variant that bridges Gaussian noise to the fine target.

    The path is defined between a random Gaussian sample and the target image,
    mirroring diffusion-style starting points while keeping the flow-matching
    training objective and trainer interface intact.
    """

    def __init__(
        self,
        model,
        loss_type='l2',
        num_steps=20,
        noise_std=1.0,
        path_schedule='linear',
        path_power=1.0,
        volume_consistency_weight=0.0,
        condition_dropout_p=0.0,
        change_loss_weight=0.0,
        coarse_consistency_weight=0.0,
        coarse_consistency_loss_type='l1',
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.path_schedule = path_schedule
        self.path_power = path_power
        self.volume_consistency_weight = volume_consistency_weight
        self.condition_dropout_p = condition_dropout_p
        self.change_loss_weight = change_loss_weight
        self.coarse_consistency_weight = coarse_consistency_weight
        self.coarse_consistency_loss_type = coarse_consistency_loss_type

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def _alpha_and_derivative(self, t):
        schedule = self.path_schedule.lower()
        if schedule in ('linear',):
            alpha = t
            alpha_prime = torch.ones_like(t)
        elif schedule in ('poly', 'polynomial'):
            alpha = t.clamp(min=0.0) ** self.path_power
            if self.path_power == 0:
                alpha_prime = torch.zeros_like(t)
            else:
                alpha_prime = self.path_power * t.clamp(min=0.0) ** max(self.path_power - 1.0, 0.0)
        elif schedule in ('cos', 'cosine'):
            alpha = 0.5 - 0.5 * torch.cos(math.pi * t)
            alpha_prime = 0.5 * math.pi * torch.sin(math.pi * t)
        else:
            raise ValueError(f'invalid path schedule {self.path_schedule}')
        return alpha, alpha_prime

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02):
        b = fine_img_02.shape[0]
        device = fine_img_02.device

        # Sample Gaussian noise as the starting point
        noise = torch.randn_like(fine_img_02) * self.noise_std

        # Sample time t
        t = torch.rand(b, device=device).type_as(fine_img_02)
        t = rearrange(t, 'b -> b 1 1 1')

        alpha, alpha_prime = self._alpha_and_derivative(t)

        # Path between noise and target
        x_t = (1 - alpha) * noise + alpha * fine_img_02 + torch.randn_like(fine_img_02) * (1 - alpha) * 0.01

        # Target vector field
        u_t = alpha_prime * (fine_img_02 - noise)

        fine_img_01_cond = maybe_apply_condition_dropout(
            fine_img_01, self.condition_dropout_p, self.training
        )
        pred_u_t = self.model(
            coarse_img_01, coarse_img_02, fine_img_01_cond, x_t, t.squeeze()
        )
        per_pixel_loss = self.loss_fn(pred_u_t, u_t, reduction='none')
        weight_map = build_change_weight_map(
            coarse_img_01,
            coarse_img_02,
            target_spatial_shape=u_t.shape[-2:],
            change_loss_weight=self.change_loss_weight,
        )
        if weight_map is not None:
            per_pixel_loss = per_pixel_loss * weight_map
        loss = per_pixel_loss.mean()

        if self.coarse_consistency_weight > 0.0:
            alpha_prime_safe = alpha_prime.clamp(min=1e-3)
            pred_fine_img_02 = noise + pred_u_t / alpha_prime_safe
            cst_loss = coarse_consistency_loss(
                pred_fine_img_02,
                coarse_img_02,
                loss_type=self.coarse_consistency_loss_type,
            )
            loss = loss + self.coarse_consistency_weight * cst_loss

        # Volume consistency loss at final state (t=1)
        if self.volume_consistency_weight > 0:
            fine_volume = compute_volume(fine_img_02)
            coarse_volume = compute_volume(coarse_img_02)
            volume_loss = F.mse_loss(fine_volume, coarse_volume)
            loss = loss + self.volume_consistency_weight * volume_loss
        
        return loss

    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        device = coarse_img_01.device
        dtype = fine_img_01.dtype
        noise = torch.randn_like(fine_img_01) * self.noise_std
        x_t = noise.clone()
        dt = 1.0 / self.num_steps

        for t_step in range(self.num_steps):
            t = torch.full(
                (fine_img_01.shape[0],),
                t_step * dt,
                device=device,
                dtype=dtype,
            )
            pred_u_t = self.model(coarse_img_01, coarse_img_02, fine_img_01, x_t, t)
            x_t = x_t + pred_u_t * dt

        return x_t


class ResidualGaussianFlowMatching(nn.Module):
    """
    Gaussian-start flow that learns the residual map: fine_img_02 - fine_img_01.
    The start distribution is a coarse-influenced Gaussian around the coarse residual.
    """

    def __init__(
        self,
        model,
        loss_type='l2',
        num_steps=20,
        noise_std=1.0,
        path_schedule='linear',
        path_power=1.0,
        coarse_weight=1.0,
        volume_consistency_weight=0.0,
        change_loss_weight=0.0,
        coarse_consistency_weight=0.0,
        coarse_consistency_loss_type='l1',
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.path_schedule = path_schedule
        self.path_power = path_power
        self.coarse_weight = coarse_weight
        self.volume_consistency_weight = volume_consistency_weight
        self.change_loss_weight = change_loss_weight
        self.coarse_consistency_weight = coarse_consistency_weight
        self.coarse_consistency_loss_type = coarse_consistency_loss_type

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def _alpha_and_derivative(self, t):
        schedule = self.path_schedule.lower()
        if schedule in ('linear',):
            alpha = t
            alpha_prime = torch.ones_like(t)
        elif schedule in ('poly', 'polynomial'):
            alpha = t.clamp(min=0.0) ** self.path_power
            if self.path_power == 0:
                alpha_prime = torch.zeros_like(t)
            else:
                alpha_prime = self.path_power * t.clamp(min=0.0) ** max(self.path_power - 1.0, 0.0)
        elif schedule in ('cos', 'cosine'):
            alpha = 0.5 - 0.5 * torch.cos(math.pi * t)
            alpha_prime = 0.5 * math.pi * torch.sin(math.pi * t)
        else:
            raise ValueError(f'invalid path schedule {self.path_schedule}')
        return alpha, alpha_prime

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02):
        b = fine_img_02.shape[0]
        device = fine_img_02.device

        # Residual target
        delta = fine_img_02 - fine_img_01
        coarse_delta = coarse_img_02 - coarse_img_01

        # Coarse-influenced Gaussian start around coarse residual
        z_mean = self.coarse_weight * coarse_delta
        z = z_mean + torch.randn_like(delta) * self.noise_std

        # Sample time
        t = torch.rand(b, device=device).type_as(delta)
        t = rearrange(t, 'b -> b 1 1 1')
        alpha, alpha_prime = self._alpha_and_derivative(t)

        # Path and target vector field in residual space
        x_t = (1 - alpha) * z + alpha * delta
        u_t = alpha_prime * (delta - z)

        pred_u_t = self.model(
            coarse_img_01, coarse_img_02, fine_img_01, x_t, t.squeeze()
        )
        per_pixel_loss = self.loss_fn(pred_u_t, u_t, reduction='none')
        weight_map = build_change_weight_map(
            coarse_img_01,
            coarse_img_02,
            target_spatial_shape=u_t.shape[-2:],
            change_loss_weight=self.change_loss_weight,
        )
        if weight_map is not None:
            per_pixel_loss = per_pixel_loss * weight_map
        loss = per_pixel_loss.mean()

        if self.coarse_consistency_weight > 0.0:
            alpha_prime_safe = alpha_prime.clamp(min=1e-3)
            pred_delta = z + pred_u_t / alpha_prime_safe
            pred_fine_img_02 = fine_img_01 + pred_delta
            cst_loss = coarse_consistency_loss(
                pred_fine_img_02,
                coarse_img_02,
                loss_type=self.coarse_consistency_loss_type,
            )
            loss = loss + self.coarse_consistency_weight * cst_loss

        # Volume consistency loss at final state (t=1)
        if self.volume_consistency_weight > 0:
            fine_volume = compute_volume(fine_img_02)
            coarse_volume = compute_volume(coarse_img_02)
            volume_loss = F.mse_loss(fine_volume, coarse_volume)
            loss = loss + self.volume_consistency_weight * volume_loss
        
        return loss

    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        device = fine_img_01.device
        dtype = fine_img_01.dtype
        dt = 1.0 / self.num_steps

        # Initialize residual state near the coarse residual
        coarse_delta = coarse_img_02 - coarse_img_01
        z_mean = self.coarse_weight * coarse_delta
        x_t = z_mean + torch.randn_like(fine_img_01) * self.noise_std

        for t_step in range(self.num_steps):
            t = torch.full(
                (fine_img_01.shape[0],),
                t_step * dt,
                device=device,
                dtype=dtype,
            )
            pred_u_t = self.model(
                coarse_img_01, coarse_img_02, fine_img_01, x_t, t
            )
            x_t = x_t + pred_u_t * dt

        # Return reconstructed fine image by adding residual to fine_img_01
        return fine_img_01 + x_t
