from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class BackendLogger:
    def __init__(self, log_dir: str | Path):
        self.writer = SummaryWriter(str(log_dir))

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def add_scalars(self, name: str, values: dict[str, float], step: int) -> None:
        self.writer.add_scalars(name, values, step)

    def close(self) -> None:
        self.writer.close()
