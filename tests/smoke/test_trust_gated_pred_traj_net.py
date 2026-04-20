import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


CHANGE_MAPS_PATH = Path(__file__).resolve().parents[2] / "stf" / "models" / "change_maps.py"
CHANGE_MAPS_SPEC = spec_from_file_location("change_maps", CHANGE_MAPS_PATH)
CHANGE_MAPS_MODULE = module_from_spec(CHANGE_MAPS_SPEC)
assert CHANGE_MAPS_SPEC is not None and CHANGE_MAPS_SPEC.loader is not None
sys.modules["change_maps"] = CHANGE_MAPS_MODULE
CHANGE_MAPS_SPEC.loader.exec_module(CHANGE_MAPS_MODULE)

PRED_RESNET_PATH = Path(__file__).resolve().parents[2] / "stf" / "models" / "pred_resnet.py"
PRED_RESNET_SPEC = spec_from_file_location("stf_models_pred_resnet_test", PRED_RESNET_PATH)
PRED_RESNET_MODULE = module_from_spec(PRED_RESNET_SPEC)
assert PRED_RESNET_SPEC is not None and PRED_RESNET_SPEC.loader is not None
PRED_RESNET_SPEC.loader.exec_module(PRED_RESNET_MODULE)

PredTrajNet = PRED_RESNET_MODULE.PredTrajNet
build_soft_change_map = CHANGE_MAPS_MODULE.build_soft_change_map
summarize_trust_by_change = CHANGE_MAPS_MODULE.summarize_trust_by_change


def test_build_soft_change_map_range_and_shape():
    coarse_01 = torch.zeros(2, 4, 8, 8)
    coarse_02 = coarse_01.clone()
    coarse_02[:, :, 2:6, 2:6] = 1.0

    change_map = build_soft_change_map(
        coarse_01,
        coarse_02,
        target_spatial_shape=(16, 16),
        smooth_kernel_size=3,
        power=1.0,
    )

    assert change_map.shape == (2, 1, 16, 16)
    assert float(change_map.min()) >= 0.0
    assert float(change_map.max()) <= 1.0


def test_pred_traj_net_without_trust_gate_has_no_observability():
    model = PredTrajNet(dim=8, channels=4, out_dim=4, dim_mults=(1, 2), trust_gate_enabled=False)
    model.eval()

    coarse_01 = torch.zeros(2, 4, 16, 16)
    coarse_02 = torch.zeros(2, 4, 16, 16)
    fine_01 = torch.zeros(2, 4, 16, 16)
    noisy = torch.zeros(2, 4, 16, 16)
    time = torch.zeros(2)

    outputs = model(coarse_01, coarse_02, fine_01, noisy, time)
    assert outputs.shape == (2, 4, 16, 16)
    assert model.get_last_trust_observability() is None


def test_pred_traj_net_trust_gate_starts_near_identity():
    model = PredTrajNet(
        dim=8,
        channels=4,
        out_dim=4,
        dim_mults=(1, 2),
        trust_gate_enabled=True,
        trust_gate_init=0.9,
    )
    model.eval()

    coarse_01 = torch.zeros(2, 4, 16, 16)
    coarse_02 = coarse_01.clone()
    coarse_02[:, :, 4:12, 4:12] = 1.0
    fine_01 = torch.zeros(2, 4, 16, 16)
    noisy = torch.zeros(2, 4, 16, 16)
    time = torch.zeros(2)

    outputs = model(coarse_01, coarse_02, fine_01, noisy, time)
    observability = model.get_last_trust_observability()

    assert outputs.shape == (2, 4, 16, 16)
    assert observability is not None
    assert observability["trust_map"].shape == (2, 1, 16, 16)
    assert 0.85 <= observability["trust_mean"] <= 0.95


def test_summarize_trust_by_change_returns_expected_keys():
    trust_map = torch.full((1, 1, 8, 8), 0.75)
    change_map = torch.zeros(1, 1, 8, 8)
    change_map[:, :, 2:6, 2:6] = 1.0

    summary = summarize_trust_by_change(trust_map, change_map)

    assert set(summary) == {"trust_mean", "trust_changed", "trust_unchanged"}
    assert summary["trust_mean"] == 0.75
    assert summary["trust_changed"] == 0.75
    assert summary["trust_unchanged"] == 0.75
