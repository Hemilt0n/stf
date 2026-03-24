from pathlib import Path
from types import SimpleNamespace

import torch

from stf.engine.train import TrainEngine


def _build_engine(max_epochs: int, val_interval: int, save_interval: int, with_val_loader: bool = True):
    engine = TrainEngine.__new__(TrainEngine)
    engine.experiment = SimpleNamespace(
        train=SimpleNamespace(
            max_epochs=max_epochs,
            val_interval=val_interval,
            save_interval=save_interval,
        )
    )
    engine.val_loader = [object()] if with_val_loader else None
    engine.scheduler = None
    engine.device = torch.device("cuda:0")
    engine.current_epoch = 0
    engine.run_dirs = {"root": Path("runs/dummy")}

    calls = {"train": 0, "val": 0, "save": 0, "close": 0, "log": 0}

    def mark_train():
        calls["train"] += 1

    def mark_val():
        calls["val"] += 1
        return {"loss": 0.0}

    def mark_save():
        calls["save"] += 1

    def mark_close():
        calls["close"] += 1

    def mark_log():
        calls["log"] += 1

    engine._run_train_epoch = mark_train
    engine._run_val_epoch = mark_val
    engine._save_checkpoint = mark_save
    engine.close = mark_close
    engine._log_peak_memory_stats = mark_log
    return engine, calls


def test_run_disable_val_and_save_when_interval_non_positive(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)

    engine, calls = _build_engine(max_epochs=3, val_interval=0, save_interval=-1, with_val_loader=True)
    engine.run()

    assert calls["train"] == 3
    assert calls["val"] == 0
    assert calls["save"] == 0
    assert calls["log"] == 1
    assert calls["close"] == 1


def test_run_keeps_positive_interval_behavior(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)

    engine, calls = _build_engine(max_epochs=5, val_interval=2, save_interval=3, with_val_loader=True)
    engine.run()

    assert calls["train"] == 5
    assert calls["val"] == 2
    assert calls["save"] == 1
