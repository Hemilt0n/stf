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
    precision: str = "fp16"
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    enable_tf32: bool = False
    deterministic: bool = True
    cudnn_benchmark: bool = False
    non_blocking_transfer: bool = False
    train_log_interval: int = 1
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    compile_dynamic: bool = False
    use_channels_last: bool = False
    fine_t1_noise_warmup_epochs: int = 0
    fine_t1_noise_warmup_steps: int = 0
    fine_t1_noise_power: float = 4.0
    fine_t1_noise_std: float = 1.0
    fine_t1_noise_alpha_tail: float = 0.0
    val_step_log_keys: bool = False
    val_step_log_max_keys: int = 8
    val_step_save_csv: bool = False
    val_trust_log_stats: bool = False
    val_trust_save_max: int = 0


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
