#!/usr/bin/env python3
"""Summarize STF train.log by epoch.

Given a `train.log`, generate compact epoch-level summaries in the same folder:
- `<basename>.epoch.csv`
- `<basename>.epoch.md`
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


LINE_RE = re.compile(
    r"^(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) - .* - INFO - (?P<msg>.*)$"
)
MSG_RE = re.compile(r"^(?P<phase>train|val) epoch=(?P<epoch>\d+)\s+(?P<body>.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+)=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")

PREFERRED_METRIC_ORDER = [
    "RMSE",
    "MAE",
    "PSNR",
    "SSIM",
    "ERGAS",
    "CC",
    "SAM",
    "UIQI",
    "TRP",
]


@dataclass
class EpochStats:
    train_iter_losses: List[float] = field(default_factory=list)
    val_iter_losses: List[float] = field(default_factory=list)
    train_avg_loss: float | None = None
    train_loss_summary: float | None = None
    val_loss_summary: float | None = None
    val_metrics: Dict[str, float] = field(default_factory=dict)


def normalize_metric_name(name: str) -> str:
    return "ERGAS" if name.lower() == "ergas" else name


def parse_kv_pairs(text: str) -> Dict[str, float]:
    return {m.group(1): float(m.group(2)) for m in KV_RE.finditer(text)}


def parse_log(log_path: Path) -> Dict[int, EpochStats]:
    per_epoch: Dict[int, EpochStats] = {}
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line_match = LINE_RE.match(raw_line)
        if not line_match:
            continue
        msg = line_match.group("msg")
        msg_match = MSG_RE.match(msg)
        if not msg_match:
            continue

        phase = msg_match.group("phase")
        epoch = int(msg_match.group("epoch"))
        body = msg_match.group("body")
        stats = per_epoch.setdefault(epoch, EpochStats())
        kv = parse_kv_pairs(body)

        if "iter" in kv and "loss" in kv:
            if phase == "train":
                stats.train_iter_losses.append(kv["loss"])
            else:
                stats.val_iter_losses.append(kv["loss"])
            continue

        if phase == "train":
            if "avg_loss" in kv:
                stats.train_avg_loss = kv["avg_loss"]
            if "loss" in kv:
                stats.train_loss_summary = kv["loss"]
            continue

        if "loss" in kv:
            stats.val_loss_summary = kv["loss"]
        for key, value in kv.items():
            if key in {"iter", "loss"}:
                continue
            stats.val_metrics[normalize_metric_name(key)] = value

    return per_epoch


def safe_mean(values: List[float]) -> float | None:
    return statistics.fmean(values) if values else None


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def sorted_metric_keys(per_epoch: Dict[int, EpochStats]) -> List[str]:
    keys = set()
    for stats in per_epoch.values():
        keys.update(stats.val_metrics.keys())
    preferred = [k for k in PREFERRED_METRIC_ORDER if k in keys]
    others = sorted(k for k in keys if k not in PREFERRED_METRIC_ORDER)
    return preferred + others


def build_rows(per_epoch: Dict[int, EpochStats], metric_keys: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for epoch in sorted(per_epoch.keys()):
        stats = per_epoch[epoch]
        train_iter_mean = safe_mean(stats.train_iter_losses)
        val_iter_mean = safe_mean(stats.val_iter_losses)
        train_loss = stats.train_avg_loss
        if train_loss is None:
            train_loss = stats.train_loss_summary
        if train_loss is None:
            train_loss = train_iter_mean
        val_loss = stats.val_loss_summary
        if val_loss is None:
            val_loss = val_iter_mean

        row = {
            "epoch": str(epoch),
            "train_iter_count": str(len(stats.train_iter_losses)),
            "train_loss_iter_mean": format_float(train_iter_mean),
            "train_avg_loss_logged": format_float(stats.train_avg_loss),
            "train_loss": format_float(train_loss),
            "val_iter_count": str(len(stats.val_iter_losses)),
            "val_loss_iter_mean": format_float(val_iter_mean),
            "val_loss_logged": format_float(stats.val_loss_summary),
            "val_loss": format_float(val_loss),
        }
        for key in metric_keys:
            row[key] = format_float(stats.val_metrics.get(key))
        rows.append(row)
    return rows


def write_csv(csv_path: Path, rows: List[Dict[str, str]], metric_keys: List[str]) -> None:
    fieldnames = [
        "epoch",
        "train_iter_count",
        "train_loss_iter_mean",
        "train_avg_loss_logged",
        "train_loss",
        "val_iter_count",
        "val_loss_iter_mean",
        "val_loss_logged",
        "val_loss",
        *metric_keys,
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: List[Dict[str, str]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r.get(c, "") for c in cols) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def select_focus_metric(metric_keys: List[str]) -> str | None:
    for name in ("TRP", "RMSE", "MAE", "SSIM"):
        if name in metric_keys:
            return name
    return metric_keys[0] if metric_keys else None


def write_markdown(md_path: Path, rows: List[Dict[str, str]], metric_keys: List[str], log_path: Path) -> None:
    focus_metric = select_focus_metric(metric_keys)
    cols = [
        "epoch",
        "train_loss",
        "val_loss",
    ]
    if focus_metric:
        cols.append(focus_metric)

    best_by_val_loss = ""
    val_loss_rows = [r for r in rows if r.get("val_loss")]
    if val_loss_rows:
        best_row = min(val_loss_rows, key=lambda r: float(r["val_loss"]))
        best_by_val_loss = (
            f"- best val_loss epoch: `{best_row['epoch']}` "
            f"(val_loss={best_row['val_loss']})"
        )

    lines = [
        f"# Epoch Summary for `{log_path.name}`",
        "",
        f"- source: `{log_path}`",
        f"- epochs: `{len(rows)}`",
        best_by_val_loss,
        "",
        "## Compact View",
        "",
        md_table(rows, cols),
        "",
        "## Full Columns",
        "",
        md_table(rows, list(rows[0].keys()) if rows else cols),
        "",
    ]
    md_path.write_text("\n".join(line for line in lines if line != ""), encoding="utf-8")


def summarize(log_path: Path) -> tuple[Path, Path, int]:
    if not log_path.exists():
        raise FileNotFoundError(f"train log not found: {log_path}")
    if log_path.is_dir():
        raise IsADirectoryError(f"expected log file, got directory: {log_path}")

    per_epoch = parse_log(log_path)
    metric_keys = sorted_metric_keys(per_epoch)
    rows = build_rows(per_epoch, metric_keys)

    out_base = log_path.with_suffix("")
    csv_path = out_base.with_suffix(".epoch.csv")
    md_path = out_base.with_suffix(".epoch.md")
    write_csv(csv_path, rows, metric_keys)
    write_markdown(md_path, rows, metric_keys, log_path)
    return csv_path, md_path, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize STF train.log into epoch-level files")
    parser.add_argument("log", type=Path, help="Path to train.log")
    args = parser.parse_args()

    csv_path, md_path, epoch_count = summarize(args.log)
    print(f"[ok] epochs={epoch_count}")
    print(f"[ok] csv={csv_path}")
    print(f"[ok] md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
