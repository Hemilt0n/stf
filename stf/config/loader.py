from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

from .types import DataConfig, ExperimentConfig, IOConfig, TrainConfig


def _load_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config_module(config: str) -> ModuleType:
    path = Path(config)
    if path.suffix == ".py" or path.exists():
        return _load_module_from_path(path.resolve())
    return importlib.import_module(config)


def _coerce_experiment(obj) -> ExperimentConfig:
    if isinstance(obj, ExperimentConfig):
        return obj
    if isinstance(obj, dict):
        data = obj.get("data")
        train = obj.get("train")
        io = obj.get("io")
        if isinstance(data, dict):
            data = DataConfig(**data)
        if isinstance(train, dict):
            train = TrainConfig(**train)
        if isinstance(io, dict):
            io = IOConfig(**io)
        return ExperimentConfig(
            task=obj["task"],
            name=obj["name"],
            model=obj["model"],
            optimizer=obj["optimizer"],
            scheduler=obj.get("scheduler"),
            metrics=obj.get("metrics", []),
            data=data or DataConfig(),
            train=train or TrainConfig(),
            io=io or IOConfig(),
            seed=obj.get("seed", 42),
            resume_from=obj.get("resume_from"),
            message=obj.get("message", ""),
            legacy=obj.get("legacy", {}),
        )
    raise TypeError("Config must export EXPERIMENT as ExperimentConfig or dict")


def load_experiment(config: str) -> ExperimentConfig:
    module = load_config_module(config)
    if not hasattr(module, "EXPERIMENT"):
        raise AttributeError(f"{config} does not define EXPERIMENT")
    exp = _coerce_experiment(getattr(module, "EXPERIMENT"))
    if exp.task not in {"stfdiff", "flow"}:
        raise ValueError(f"Unsupported task: {exp.task}")
    return exp
