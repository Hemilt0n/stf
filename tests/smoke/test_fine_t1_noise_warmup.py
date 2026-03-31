from types import SimpleNamespace

import torch

from stf.engine.train import TrainEngine


def test_compute_fine_t1_noise_alpha_schedule():
    alpha0 = TrainEngine._compute_fine_t1_noise_alpha(global_step=0, warmup_steps=100, power=4.0)
    alpha_mid = TrainEngine._compute_fine_t1_noise_alpha(global_step=50, warmup_steps=100, power=4.0)
    alpha_end = TrainEngine._compute_fine_t1_noise_alpha(global_step=100, warmup_steps=100, power=4.0)

    assert alpha0 == 1.0
    assert 0.0 < alpha_mid < alpha0
    assert alpha_end == 0.0


def test_compute_fine_t1_noise_alpha_schedule_with_tail():
    alpha0 = TrainEngine._compute_fine_t1_noise_alpha(
        global_step=0, warmup_steps=100, power=4.0, alpha_tail=0.2
    )
    alpha_mid = TrainEngine._compute_fine_t1_noise_alpha(
        global_step=50, warmup_steps=100, power=4.0, alpha_tail=0.2
    )
    alpha_end = TrainEngine._compute_fine_t1_noise_alpha(
        global_step=100, warmup_steps=100, power=4.0, alpha_tail=0.2
    )
    alpha_after = TrainEngine._compute_fine_t1_noise_alpha(
        global_step=1000, warmup_steps=100, power=4.0, alpha_tail=0.2
    )

    assert alpha0 == 1.0
    assert 0.2 < alpha_mid < alpha0
    assert alpha_end == 0.2
    assert alpha_after == 0.2


def test_prepare_train_inputs_replaces_fine_t1_during_warmup():
    engine = TrainEngine.__new__(TrainEngine)
    engine.device = torch.device("cpu")
    engine.non_blocking_transfer = False
    engine.use_channels_last = False
    engine.fine_t1_noise_total_steps = 10
    engine.fine_t1_noise_power = 4.0
    engine.fine_t1_noise_std = 1.0
    engine.fine_t1_noise_alpha_tail = 0.0
    engine.current_train_step = 0
    engine.backend_logger = SimpleNamespace(add_scalar=lambda *_args, **_kwargs: None)

    batch = {
        "coarse_img_01": torch.zeros(2, 3, 8, 8),
        "coarse_img_02": torch.zeros(2, 3, 8, 8),
        "fine_img_01": torch.zeros(2, 3, 8, 8),
        "fine_img_02": torch.zeros(2, 3, 8, 8),
    }

    torch.manual_seed(0)
    _coarse1, _coarse2, fine_t1_warmup, _fine_t2 = engine._prepare_train_inputs(batch)
    assert not torch.allclose(fine_t1_warmup, batch["fine_img_01"])

    engine.current_train_step = 10
    torch.manual_seed(0)
    _coarse1, _coarse2, fine_t1_after, _fine_t2 = engine._prepare_train_inputs(batch)
    assert torch.allclose(fine_t1_after, batch["fine_img_01"])


def test_prepare_train_inputs_keeps_tail_after_warmup():
    engine = TrainEngine.__new__(TrainEngine)
    engine.device = torch.device("cpu")
    engine.non_blocking_transfer = False
    engine.use_channels_last = False
    engine.fine_t1_noise_total_steps = 10
    engine.fine_t1_noise_power = 4.0
    engine.fine_t1_noise_std = 1.0
    engine.fine_t1_noise_alpha_tail = 0.2
    engine.current_train_step = 10
    engine.backend_logger = SimpleNamespace(add_scalar=lambda *_args, **_kwargs: None)

    batch = {
        "coarse_img_01": torch.zeros(2, 3, 8, 8),
        "coarse_img_02": torch.zeros(2, 3, 8, 8),
        "fine_img_01": torch.zeros(2, 3, 8, 8),
        "fine_img_02": torch.zeros(2, 3, 8, 8),
    }

    torch.manual_seed(0)
    _coarse1, _coarse2, fine_t1_after, _fine_t2 = engine._prepare_train_inputs(batch)
    assert not torch.allclose(fine_t1_after, batch["fine_img_01"])
