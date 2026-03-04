from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_run_dirs(base_dir: Path, with_timestamp: bool = True) -> dict[str, Path]:
    run_root = Path(base_dir)
    if with_timestamp:
        run_root = run_root.parent / f"{run_root.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    dirs = {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "logs": run_root / "logs",
        "tensorboard": run_root / "tensorboard",
        "images": run_root / "images",
        "configs": run_root / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
