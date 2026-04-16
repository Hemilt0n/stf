from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


FLOW_PATH = Path(__file__).resolve().parents[2] / "stf" / "models" / "flow.py"
FLOW_SPEC = spec_from_file_location("stf_models_flow_test", FLOW_PATH)
FLOW_MODULE = module_from_spec(FLOW_SPEC)
assert FLOW_SPEC is not None and FLOW_SPEC.loader is not None
FLOW_SPEC.loader.exec_module(FLOW_MODULE)

ResidualGaussianFlowMatching = FLOW_MODULE.ResidualGaussianFlowMatching
build_soft_change_map = FLOW_MODULE.build_soft_change_map


def test_build_soft_change_map_returns_zero_for_no_change():
    coarse_img_01 = torch.zeros(2, 3, 8, 8)
    coarse_img_02 = torch.zeros(2, 3, 8, 8)

    change_map = build_soft_change_map(
        coarse_img_01,
        coarse_img_02,
        target_spatial_shape=(8, 8),
        smooth_kernel_size=1,
        power=1.0,
    )

    assert change_map.shape == (2, 1, 8, 8)
    assert torch.allclose(change_map, torch.zeros_like(change_map))


def test_build_soft_change_map_highlights_changed_region():
    coarse_img_01 = torch.zeros(1, 3, 8, 8)
    coarse_img_02 = torch.zeros(1, 3, 8, 8)
    coarse_img_02[:, :, 4, 4] = 2.0

    change_map = build_soft_change_map(
        coarse_img_01,
        coarse_img_02,
        target_spatial_shape=(8, 8),
        smooth_kernel_size=1,
        power=1.0,
    )

    changed = change_map[0, 0, 4, 4].item()
    unchanged = change_map[0, 0, 0, 0].item()

    assert 0.0 <= unchanged <= 1.0
    assert 0.0 <= changed <= 1.0
    assert changed > unchanged


def test_residual_flow_start_distribution_matches_legacy_behavior_when_disabled():
    model = ResidualGaussianFlowMatching(
        model=torch.nn.Identity(),
        noise_std=0.7,
        coarse_weight=1.5,
        geo_edit_enabled=False,
    )
    coarse_img_01 = torch.zeros(1, 3, 8, 8)
    coarse_img_02 = torch.randn(1, 3, 8, 8)

    z_mean, sigma, edit_mask = model._build_residual_start_distribution(
        coarse_img_01,
        coarse_img_02,
        target_spatial_shape=(8, 8),
    )

    assert edit_mask is None
    assert torch.allclose(z_mean, 1.5 * (coarse_img_02 - coarse_img_01))
    assert torch.allclose(sigma, torch.full_like(z_mean, 0.7))


def test_residual_flow_start_distribution_uses_spatially_varying_geo_edit_path():
    model = ResidualGaussianFlowMatching(
        model=torch.nn.Identity(),
        coarse_weight=1.0,
        geo_edit_enabled=True,
        geo_edit_sigma_low=0.1,
        geo_edit_sigma_high=0.9,
        geo_edit_mask_power=1.0,
        geo_edit_mask_smooth_kernel=1,
    )
    coarse_img_01 = torch.zeros(1, 3, 8, 8)
    coarse_img_02 = torch.zeros(1, 3, 8, 8)
    coarse_img_02[:, :, 4, 4] = 2.0

    z_mean, sigma, edit_mask = model._build_residual_start_distribution(
        coarse_img_01,
        coarse_img_02,
        target_spatial_shape=(8, 8),
    )

    assert edit_mask is not None
    assert z_mean.shape == coarse_img_01.shape
    assert sigma.shape == coarse_img_01.shape

    changed_sigma = sigma[0, 0, 4, 4].item()
    unchanged_sigma = sigma[0, 0, 0, 0].item()
    changed_mean = z_mean[0, 0, 4, 4].item()
    unchanged_mean = z_mean[0, 0, 0, 0].item()

    assert changed_sigma > unchanged_sigma
    assert unchanged_sigma == pytest.approx(0.1)
    assert changed_mean > unchanged_mean
    assert unchanged_mean == pytest.approx(0.0)
