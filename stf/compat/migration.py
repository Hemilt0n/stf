from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path


COMMON_FIELDS = [
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "model",
    "optimizer",
    "scheduler",
    "metric_list",
    "MAX_EPOCH",
    "VAL_INTERVAL",
    "SAVE_INTERVAL",
    "show_bands",
    "is_save_img",
    "is_show_img",
    "checkpoint_path",
]


def infer_task_from_legacy_path(path: str | Path) -> str:
    p = str(path).lower()
    return "flow" if "flow" in p else "stfdiff"


def _read_assigned_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def load_legacy_config_objects(path: str | Path) -> dict:
    path = Path(path).resolve()
    repo_root = path.parent
    while repo_root != repo_root.parent:
        if (repo_root / "src").exists():
            break
        repo_root = repo_root.parent

    sys.path.insert(0, str(repo_root))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import legacy config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))

    data = {}
    for field in COMMON_FIELDS:
        if hasattr(module, field):
            data[field] = getattr(module, field)
    return data


def migrate_legacy_config(legacy_config: str | Path, output_path: str | Path) -> dict:
    legacy_config = Path(legacy_config).resolve()
    output_path = Path(output_path).resolve()
    task = infer_task_from_legacy_path(legacy_config)
    names = _read_assigned_names(legacy_config)

    template = f'''from stf.compat.migration import load_legacy_config_objects\nfrom stf.config.types import DataConfig, ExperimentConfig, IOConfig, TrainConfig\n\nlegacy = load_legacy_config_objects("{legacy_config}")\n\nEXPERIMENT = ExperimentConfig(\n    task="{task}",\n    name="{legacy_config.stem}",\n    model=legacy["model"],\n    optimizer=legacy["optimizer"],\n    scheduler=legacy.get("scheduler"),\n    metrics=legacy.get("metric_list", []),\n    data=DataConfig(\n        train_dataloader=legacy.get("train_dataloader"),\n        val_dataloader=legacy.get("val_dataloader"),\n        test_dataloader=legacy.get("test_dataloader"),\n    ),\n    train=TrainConfig(\n        max_epochs=legacy.get("MAX_EPOCH", 1),\n        val_interval=legacy.get("VAL_INTERVAL", 1),\n        save_interval=legacy.get("SAVE_INTERVAL", 1),\n        use_ema=True,\n        use_mixed_precision=True,\n    ),\n    io=IOConfig(\n        output_root="runs",\n        save_images=legacy.get("is_save_img", False),\n        show_images=legacy.get("is_show_img", False),\n        show_bands=legacy.get("show_bands", (2, 1, 0)),\n    ),\n    resume_from=legacy.get("checkpoint_path"),\n    legacy={{"source_config": "{legacy_config}"}},\n)\n'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template)
    return {
        "legacy_config": str(legacy_config),
        "output_path": str(output_path),
        "task": task,
        "detected_fields": sorted(names.intersection(COMMON_FIELDS)),
        "missing_common_fields": sorted(set(COMMON_FIELDS) - names),
    }
