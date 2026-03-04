from __future__ import annotations

from pathlib import Path

from stf.compat import migrate_legacy_config
from stf.config import load_experiment
from stf.engine import EvalEngine, PredictEngine, TrainEngine


def train(config: str, output_dir: str | None = None, resume_from: str | None = None):
    exp = load_experiment(config)
    engine = TrainEngine(exp, config_path=config, output_dir=output_dir, resume_from=resume_from)
    return engine.run()


def evaluate(config: str, checkpoint: str, output_dir: str | None = None):
    exp = load_experiment(config)
    engine = EvalEngine(exp, config_path=config, checkpoint_path=checkpoint, output_dir=output_dir)
    return engine.run()


def predict(config: str, checkpoint: str, output_dir: str | None = None):
    exp = load_experiment(config)
    engine = PredictEngine(exp, config_path=config, checkpoint_path=checkpoint, output_dir=output_dir)
    return engine.run()


def migrate_config(legacy_config: str, output: str) -> dict:
    return migrate_legacy_config(legacy_config, output)
