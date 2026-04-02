#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _normalize_suffix(suffix: str) -> str:
    suffix = suffix.strip().lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def _iter_source_files(input_root: Path, source_suffix: str):
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == source_suffix:
            yield path


def _save_array(dst_path: Path, array: np.ndarray, fmt: str) -> None:
    if fmt == "npy":
        np.save(dst_path, array, allow_pickle=False)
        return
    if fmt == "npz":
        np.savez_compressed(dst_path, array=array)
        return
    raise ValueError(f"Unsupported output format: {fmt}")


def _derive_output_root(input_root: Path, output_root: str | None) -> Path:
    if output_root:
        return Path(output_root).resolve()

    split_name = input_root.name.lower()
    if split_name in {"train", "val", "test"}:
        dataset_root = input_root.parent
        serialized_dataset_root = dataset_root.parent / f"{dataset_root.name}_serialized"
        return serialized_dataset_root / input_root.name

    return input_root.parent / f"{input_root.name}_serialized"


def _serialize_single_root(
    input_root: Path,
    output_root: Path,
    source_suffix: str,
    array_format: str,
    overwrite: bool,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = list(_iter_source_files(input_root, source_suffix))
    total = len(source_files)
    converted = 0
    skipped = 0
    target_suffix = f".{array_format}"

    use_tqdm = tqdm is not None
    iterable = source_files
    if use_tqdm:
        iterable = tqdm(source_files, desc=f"serialize:{input_root.name}", unit="file")
    else:
        print(
            "[warn] tqdm is not installed; using plain progress logs. "
            "Install tqdm for a live progress bar."
        )

    for idx, src_path in enumerate(iterable, start=1):
        rel = src_path.relative_to(input_root)
        dst_path = (output_root / rel).with_suffix(target_suffix)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not overwrite:
            skipped += 1
            if use_tqdm:
                iterable.set_postfix(converted=converted, skipped=skipped, refresh=False)
            continue

        array = tifffile.imread(src_path)
        _save_array(dst_path, array, array_format)
        converted += 1

        if use_tqdm:
            iterable.set_postfix(converted=converted, skipped=skipped, refresh=False)
        elif idx % 200 == 0 or idx == total:
            print(
                "[progress] "
                f"input={input_root} output={output_root} "
                f"converted={converted} skipped={skipped} seen={idx}/{total}"
            )

    marker = {
        "format": "stf_serialized_dataset",
        "version": 1,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_root": str(input_root),
        "source_suffix": source_suffix,
        "data_suffix": target_suffix,
        "serializer": "numpy",
        "array_format": array_format,
        "total_seen": total,
        "converted": converted,
        "skipped": skipped,
    }
    marker_path = output_root / ".stf_serialized.json"
    with marker_path.open("w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, ensure_ascii=False)

    print(
        f"[ok] input={input_root} output={output_root} "
        f"seen={total} converted={converted} skipped={skipped} marker={marker_path}"
    )
    return marker


def _normalize_splits_arg(splits_arg: str | None) -> list[str]:
    if not splits_arg:
        return []
    splits = [item.strip() for item in splits_arg.split(",") if item.strip()]
    seen = set()
    out = []
    for split in splits:
        key = split.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(split)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serialize raw raster files into numpy arrays while preserving directory structure."
    )
    parser.add_argument("--input-root", required=True, help="Raw dataset root (e.g. data/CIA/train)")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Serialized dataset root. "
            "If omitted, defaults to sibling '<input>_serialized' convention."
        ),
    )
    parser.add_argument(
        "--splits",
        default="",
        help=(
            "Optional comma-separated splits (for example: train,val,test). "
            "When provided, input-root is treated as dataset root and each split is converted "
            "to '<dataset>_serialized/<split>' by default."
        ),
    )
    parser.add_argument(
        "--source-suffix",
        default=".tif",
        help="Source raster suffix to convert (default: .tif)",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "npz"),
        default="npy",
        help="Serialized array format (default: npy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing serialized files in output root",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = _derive_output_root(input_root, args.output_root)
    source_suffix = _normalize_suffix(args.source_suffix)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    splits = _normalize_splits_arg(args.splits)
    if not splits:
        _serialize_single_root(
            input_root=input_root,
            output_root=output_root,
            source_suffix=source_suffix,
            array_format=args.format,
            overwrite=args.overwrite,
        )
        return 0

    if args.output_root:
        output_dataset_root = Path(args.output_root).resolve()
    else:
        output_dataset_root = input_root.parent / f"{input_root.name}_serialized"

    for split in splits:
        split_input = input_root / split
        if not split_input.exists():
            raise FileNotFoundError(f"Split input root not found: {split_input}")
        split_output = output_dataset_root / split
        _serialize_single_root(
            input_root=split_input,
            output_root=split_output,
            source_suffix=source_suffix,
            array_format=args.format,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
