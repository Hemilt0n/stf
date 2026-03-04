from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    train_dataloader: Any = None
    val_dataloader: Any = None
    test_dataloader: Any = None


@dataclass
class TrainConfig:
    max_epochs: int = 1
    val_interval: int = 1
    save_interval: int = 1
    use_ema: bool = True
    use_mixed_precision: bool = True
    grad_clip_norm: float = 1.0


@dataclass
class IOConfig:
    output_root: str = "runs"
    save_images: bool = False
    show_images: bool = False
    show_bands: tuple[int, int, int] = (2, 1, 0)


@dataclass
class ExperimentConfig:
    task: str
    name: str
    model: Any
    optimizer: Any
    scheduler: Any = None
    metrics: list[Any] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    io: IOConfig = field(default_factory=IOConfig)
    seed: int = 42
    resume_from: str | None = None
    message: str = ""
    legacy: dict[str, Any] = field(default_factory=dict)

    def default_run_base(self) -> Path:
        return Path(self.io.output_root) / self.task / self.name
