import torch

from stf.models import GaussianFlowMatching, PredTrajNet, ResidualGaussianFlowMatching
from stf.models.hf_losses import (
    build_change_mask,
    gradient_l1_loss,
    laplacian_pyramid_l1_loss,
    ranking_prefer_target_loss,
)


def test_build_change_mask_quantile_is_binary():
    coarse_01 = torch.zeros((1, 3, 4, 4))
    coarse_02 = coarse_01.clone()
    coarse_02[:, :, :2, :2] = 1.0
    mask = build_change_mask(
        coarse_01,
        coarse_02,
        target_spatial_shape=(8, 8),
        strategy="quantile",
        quantile=0.8,
    )
    assert mask.shape == (1, 1, 8, 8)
    assert torch.all((mask == 0) | (mask == 1))


def test_gradient_and_laplacian_loss_zero_for_identical_tensors():
    pred = torch.randn((2, 3, 8, 8))
    mask = torch.ones((2, 1, 8, 8))
    grad_loss = gradient_l1_loss(pred, pred, mask=mask)
    lap_loss = laplacian_pyramid_l1_loss(pred, pred, mask=mask, num_scales=2)
    assert grad_loss.item() == 0.0
    assert lap_loss.item() == 0.0


def test_ranking_loss_prefers_target_like_prediction():
    ref = torch.zeros((1, 3, 8, 8))
    tgt = torch.ones((1, 3, 8, 8))
    pred_like_tgt = torch.ones((1, 3, 8, 8)) * 0.9
    pred_like_ref = torch.ones((1, 3, 8, 8)) * 0.1
    mask = torch.ones((1, 1, 8, 8))

    loss_like_tgt = ranking_prefer_target_loss(
        pred_like_tgt, tgt, ref, mask=mask, margin=0.02
    )
    loss_like_ref = ranking_prefer_target_loss(
        pred_like_ref, tgt, ref, mask=mask, margin=0.02
    )
    assert loss_like_tgt.item() < loss_like_ref.item()


def test_gaussian_flow_matching_with_hf_losses_runs():
    model = GaussianFlowMatching(
        model=PredTrajNet(dim=32, channels=3, out_dim=3, dim_mults=(1, 2)),
        num_steps=4,
        grad_loss_weight=0.1,
        lap_loss_weight=0.05,
        ranking_loss_weight=0.05,
        ranking_margin=0.02,
    )
    coarse_01 = torch.randn((2, 3, 16, 16))
    coarse_02 = torch.randn((2, 3, 16, 16))
    fine_01 = torch.randn((2, 3, 16, 16))
    fine_02 = torch.randn((2, 3, 16, 16))
    loss = model(coarse_01, coarse_02, fine_01, fine_02)
    assert torch.isfinite(loss)


def test_residual_gaussian_flow_matching_with_hf_losses_runs():
    model = ResidualGaussianFlowMatching(
        model=PredTrajNet(dim=32, channels=3, out_dim=3, dim_mults=(1, 2)),
        num_steps=4,
        grad_loss_weight=0.1,
        lap_loss_weight=0.05,
        ranking_loss_weight=0.05,
        ranking_margin=0.02,
    )
    coarse_01 = torch.randn((2, 3, 16, 16))
    coarse_02 = torch.randn((2, 3, 16, 16))
    fine_01 = torch.randn((2, 3, 16, 16))
    fine_02 = torch.randn((2, 3, 16, 16))
    loss = model(coarse_01, coarse_02, fine_01, fine_02)
    assert torch.isfinite(loss)
