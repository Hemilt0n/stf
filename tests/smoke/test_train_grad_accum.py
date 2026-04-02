from types import SimpleNamespace

import torch

from stf.engine.train import TrainEngine


class _DummyScaler:
    def is_enabled(self) -> bool:
        return False

    def scale(self, value):
        return value

    def unscale_(self, _optimizer):
        return None

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        return None


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float):
        super().__init__(params=params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (self.weight * x).mean()


def _build_engine_for_grad_accum(grad_accum_steps: int, num_iters: int, lr: float = 0.1):
    engine = TrainEngine.__new__(TrainEngine)
    engine.model = _ToyModel()
    engine.optimizer = _CountingSGD(engine.model.parameters(), lr=lr)
    engine.scaler = _DummyScaler()
    engine.ema = None
    engine.backend_logger = SimpleNamespace(add_scalar=lambda *_args, **_kwargs: None)
    engine.txt_logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    engine.amp_enabled = False
    engine.amp_dtype = torch.float16
    engine.train_log_interval = 1000
    engine.current_epoch = 0
    engine.current_train_step = 0
    engine.grad_accum_steps = grad_accum_steps
    engine.fine_t1_noise_total_steps = 0
    engine._last_fine_t1_noise_alpha = 0.0
    engine.experiment = SimpleNamespace(train=SimpleNamespace(grad_clip_norm=1.0))
    engine.train_loader = [{"x": torch.ones(2, 2)} for _ in range(num_iters)]
    engine._prepare_train_inputs = lambda batch: [batch["x"]]
    return engine


def _run_reference_old_style(num_iters: int, lr: float = 0.1) -> float:
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 2))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return float(model.weight.item())


def test_grad_accum_steps_one_matches_old_style_update():
    num_iters = 5
    expected_weight = _run_reference_old_style(num_iters=num_iters, lr=0.1)

    engine = _build_engine_for_grad_accum(grad_accum_steps=1, num_iters=num_iters, lr=0.1)
    engine._run_train_epoch()

    assert engine.optimizer.step_calls == num_iters
    assert engine.current_train_step == num_iters
    assert abs(float(engine.model.weight.item()) - expected_weight) < 1e-8


def test_grad_accum_steps_reduces_optimizer_step_frequency():
    num_iters = 5
    engine = _build_engine_for_grad_accum(grad_accum_steps=2, num_iters=num_iters, lr=0.1)
    engine._run_train_epoch()

    assert engine.optimizer.step_calls == 3  # ceil(5 / 2)
    assert engine.current_train_step == num_iters
