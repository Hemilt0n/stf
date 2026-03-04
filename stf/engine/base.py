from __future__ import annotations

import shutil
from pathlib import Path

import torch

from stf.io import build_run_dirs
from stf.logging import BackendLogger, FusionLogger
from stf.utils import fix_random_seed


class BaseEngine:
    def __init__(self, experiment, config_path: str, output_dir: str | None, mode: str):
        self.experiment = experiment
        self.config_path = config_path
        self.mode = mode

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this project configuration")

        self.device = torch.device("cuda")
        fix_random_seed(experiment.seed)

        base_dir = Path(output_dir) if output_dir else experiment.default_run_base()
        self.run_dirs = build_run_dirs(base_dir, with_timestamp=True)

        config_src = Path(config_path)
        if config_src.exists():
            shutil.copy(config_src, self.run_dirs["configs"] / config_src.name)

        self.txt_logger = FusionLogger(
            logger_name=f"stf.{mode}",
            log_file=self.run_dirs["logs"] / f"{mode}.log",
            log_level="INFO",
        )
        self.backend_logger = BackendLogger(self.run_dirs["tensorboard"])

        self.model = experiment.model.to(self.device)
        self.metrics = []
        for metric in experiment.metrics:
            if hasattr(metric, "to"):
                metric = metric.to(self.device)
            self.metrics.append(metric)

    def close(self) -> None:
        self.backend_logger.close()
