#!/usr/bin/env python3
"""Export TensorBoard scalar summaries to compact CSV/Markdown files.

Usage:
    uv run python tools/export_tb_scalars.py <run_dir_or_tb_dir_or_event_file>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tensorboard is required. Run with uv env: "
        "`uv run python tools/export_tb_scalars.py ...`"
    ) from exc


def find_event_files(path: Path) -> List[Path]:
    if path.is_file():
        if "events.out.tfevents." in path.name:
            return [path]
        raise FileNotFoundError(f"not a TensorBoard event file: {path}")
    files = sorted(path.rglob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"no TensorBoard event files found under: {path}")
    return files


def choose_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent
    return input_path


def load_scalars(event_files: List[Path]) -> Tuple[Dict[str, Dict[int, float]], Dict[str, List[float]]]:
    # tag_to_step_values: keep latest wall_time value per step.
    tag_to_step_values: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    tag_to_all_values: Dict[str, List[float]] = defaultdict(list)

    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        for tag in scalar_tags:
            for item in accumulator.Scalars(tag):
                tag_to_all_values[tag].append(float(item.value))
                prev = tag_to_step_values[tag].get(int(item.step))
                if prev is None or float(item.wall_time) >= prev[0]:
                    tag_to_step_values[tag][int(item.step)] = (
                        float(item.wall_time),
                        float(item.value),
                    )

    # Drop wall_time for downstream writing.
    compact = {
        tag: {step: val for step, (_wall, val) in step_map.items()}
        for tag, step_map in tag_to_step_values.items()
    }
    return compact, tag_to_all_values


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_step_csv(step_csv_path: Path, tag_to_step_values: Dict[str, Dict[int, float]]) -> None:
    tags = sorted(tag_to_step_values.keys())
    steps = sorted({step for step_map in tag_to_step_values.values() for step in step_map.keys()})
    with step_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *tags])
        for step in steps:
            row = [step]
            for tag in tags:
                row.append(format_float(tag_to_step_values[tag].get(step)))
            writer.writerow(row)


def write_summary_csv(
    summary_csv_path: Path,
    tag_to_step_values: Dict[str, Dict[int, float]],
    tag_to_all_values: Dict[str, List[float]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for tag in sorted(tag_to_step_values.keys()):
        step_values = tag_to_step_values[tag]
        all_values = tag_to_all_values.get(tag, [])
        if not step_values:
            continue
        last_step = max(step_values.keys())
        last_value = step_values[last_step]
        row = {
            "tag": tag,
            "count": str(len(all_values)),
            "first_step": str(min(step_values.keys())),
            "last_step": str(last_step),
            "last_value": format_float(last_value),
            "min": format_float(min(all_values) if all_values else None),
            "max": format_float(max(all_values) if all_values else None),
            "mean": format_float(statistics.fmean(all_values) if all_values else None),
        }
        rows.append(row)

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "count", "first_step", "last_step", "last_value", "min", "max", "mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary_md(summary_md_path: Path, source_path: Path, event_files: List[Path], rows: List[Dict[str, str]]) -> None:
    lines = [
        "# TensorBoard Scalars Summary",
        "",
        f"- source: `{source_path}`",
        f"- event_files: `{len(event_files)}`",
        f"- scalar_tags: `{len(rows)}`",
        "",
    ]
    if rows:
        cols = ["tag", "count", "first_step", "last_step", "last_value", "min", "max", "mean"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
        lines.extend([header, sep, *body, ""])
    else:
        lines.append("No scalar tags found.")

    summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def export(input_path: Path) -> Tuple[Path, Path, Path, int, int]:
    event_files = find_event_files(input_path)
    out_dir = choose_output_dir(input_path)
    step_csv_path = out_dir / "tb.scalars.step.csv"
    summary_csv_path = out_dir / "tb.scalars.summary.csv"
    summary_md_path = out_dir / "tb.scalars.summary.md"

    tag_to_step_values, tag_to_all_values = load_scalars(event_files)
    write_step_csv(step_csv_path, tag_to_step_values)
    rows = write_summary_csv(summary_csv_path, tag_to_step_values, tag_to_all_values)
    write_summary_md(summary_md_path, input_path, event_files, rows)
    total_steps = len({s for values in tag_to_step_values.values() for s in values.keys()})
    return step_csv_path, summary_csv_path, summary_md_path, len(rows), total_steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar summaries")
    parser.add_argument("path", type=Path, help="Run dir, tensorboard dir, or event file path")
    args = parser.parse_args()

    step_csv, summary_csv, summary_md, tag_count, step_count = export(args.path)
    print(f"[ok] tags={tag_count} steps={step_count}")
    print(f"[ok] step_csv={step_csv}")
    print(f"[ok] summary_csv={summary_csv}")
    print(f"[ok] summary_md={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
