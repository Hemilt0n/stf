#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from stf.config import load_experiment


def _pick_loader(exp, split: str):
    if split == "train":
        return exp.data.train_dataloader
    if split == "val":
        return exp.data.val_dataloader
    if split == "test":
        return exp.data.test_dataloader
    raise ValueError(f"Unsupported split: {split}")


def _set_sampler_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    inner_sampler = getattr(batch_sampler, "sampler", None)
    if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
        inner_sampler.set_epoch(epoch)


def _iter_batch_indices(loader) -> Iterable[list[int]]:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
        for batch in batch_sampler:
            yield [int(idx) for idx in batch]
        return

    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        raise RuntimeError("Loader has neither batch_sampler nor sampler")
    batch_size = int(getattr(loader, "batch_size", 1) or 1)
    drop_last = bool(getattr(loader, "drop_last", False))
    batch = []
    for idx in sampler:
        batch.append(int(idx))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not drop_last:
        yield batch


def _resolve_dataset_index_and_key(dataset, loader_index: int) -> tuple[int, str]:
    ds = dataset
    index = int(loader_index)
    while hasattr(ds, "dataset") and hasattr(ds, "indices"):
        index = int(ds.indices[index])
        ds = ds.dataset

    key = ""
    data_path_list = getattr(ds, "data_path_list", None)
    if data_path_list is not None and 0 <= index < len(data_path_list):
        item = data_path_list[index]
        if isinstance(item, dict):
            key = str(item.get("key", ""))
    return index, key


def _default_output_path(experiment_name: str, split: str, epoch: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs/debug") / f"{experiment_name}_{split}_sampler_epoch{epoch}_{stamp}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump sampler-derived batch layout (without loading image tensors)."
    )
    parser.add_argument("--config", "-c", required=True, help="Path or module path to experiment config")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Which dataloader split to inspect",
    )
    parser.add_argument("--epoch", type=int, default=0, help="Sampler epoch to reproduce")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: runs/debug/<name>_<split>_sampler_epoch<epoch>_<timestamp>.csv",
    )
    args = parser.parse_args()

    exp = load_experiment(args.config)
    loader = _pick_loader(exp, args.split)
    if loader is None:
        raise ValueError(f"{args.split}_dataloader is None in config: {args.config}")

    _set_sampler_epoch(loader, args.epoch)
    batches = list(_iter_batch_indices(loader))
    total_samples = sum(len(batch) for batch in batches)

    output_path = Path(args.output) if args.output else _default_output_path(exp.name, args.split, args.epoch)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "epoch",
                "batch_idx",
                "pos_in_batch",
                "global_pos",
                "relative_pos",
                "loader_index",
                "dataset_index",
                "key",
            ],
        )
        writer.writeheader()

        global_pos = 0
        for batch_idx, batch_indices in enumerate(batches):
            for pos_in_batch, loader_index in enumerate(batch_indices):
                dataset_index, key = _resolve_dataset_index_and_key(loader.dataset, loader_index)
                relative_pos = (global_pos / total_samples) if total_samples > 0 else 0.0
                writer.writerow(
                    {
                        "split": args.split,
                        "epoch": args.epoch,
                        "batch_idx": batch_idx,
                        "pos_in_batch": pos_in_batch,
                        "global_pos": global_pos,
                        "relative_pos": f"{relative_pos:.6f}",
                        "loader_index": loader_index,
                        "dataset_index": dataset_index,
                        "key": key,
                    }
                )
                global_pos += 1

    print(
        f"[ok] output={output_path} batches={len(batches)} samples={total_samples} "
        f"split={args.split} epoch={args.epoch}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
