from __future__ import annotations

from pathlib import Path

from stf.engine.eval import EvalEngine


class PredictEngine(EvalEngine):
    """Predict shares inference path with EvalEngine."""

    def run(self) -> Path:
        run_dir, _ = super().run()
        return run_dir
