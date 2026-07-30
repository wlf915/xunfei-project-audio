#!/usr/bin/env python3
"""Rebuild loss CSV tables from the published training logs.

This script uses only the Python standard library. It intentionally treats
training loss as an optimization trace, not as an audio-quality score.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+"
    r"Loss:\s+([0-9.eE+-]+)(?:\s+\|\s+LR:\s+([0-9.eE+-]+))?"
)

RUNS = [
    ("round1_5epoch", "lr1e-6", 1e-6, "constant", 5),
    ("round1_5epoch", "lr2e-6", 2e-6, "constant", 5),
    ("round1_5epoch", "lr5e-6", 5e-6, "constant", 5),
    ("round2_8epoch_cosine", "lr5e-7", 5e-7, "cosine", 8),
    ("round2_8epoch_cosine", "lr1e-6", 1e-6, "cosine", 8),
    ("round2_8epoch_cosine", "lr1.5e-6", 1.5e-6, "cosine", 8),
]


def main() -> None:
    records: list[dict[str, object]] = []

    for round_name, run_id, configured_lr, scheduler, epochs in RUNS:
        log_path = ROOT / "experiments" / round_name / run_id / "train.log"
        run_records: list[dict[str, object]] = []

        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PATTERN.search(line)
            if not match:
                continue
            epoch_zero_based = int(match.group(1))
            record = {
                "round": round_name,
                "run_id": run_id,
                "epoch": epoch_zero_based + 1,
                "epoch_zero_based": epoch_zero_based,
                "step": int(match.group(2)),
                "loss": float(match.group(3)),
                "learning_rate": (
                    float(match.group(4)) if match.group(4) else configured_lr
                ),
                "scheduler": scheduler,
            }
            run_records.append(record)
            records.append(record)

        expected = epochs * 90
        if len(run_records) != expected:
            raise RuntimeError(
                f"{round_name}/{run_id}: expected {expected} loss rows, "
                f"found {len(run_records)}"
            )

    metrics_dir = ROOT / "metrics"
    write_csv(metrics_dir / "loss_steps.csv", records)

    summaries: list[dict[str, object]] = []
    for round_name, run_id, configured_lr, scheduler, epochs in RUNS:
        for epoch in range(1, epochs + 1):
            epoch_records = [
                row
                for row in records
                if row["round"] == round_name
                and row["run_id"] == run_id
                and row["epoch"] == epoch
            ]
            losses = [float(row["loss"]) for row in epoch_records]
            summaries.append(
                {
                    "round": round_name,
                    "run_id": run_id,
                    "learning_rate": configured_lr,
                    "scheduler": scheduler,
                    "epoch": epoch,
                    "observations": len(losses),
                    "mean_loss": round(statistics.fmean(losses), 6),
                    "standard_deviation": round(statistics.pstdev(losses), 6),
                    "minimum_loss": round(min(losses), 6),
                    "maximum_loss": round(max(losses), 6),
                    "first_loss": round(losses[0], 6),
                    "last_loss": round(losses[-1], 6),
                }
            )

    write_csv(metrics_dir / "loss_epoch_summary.csv", summaries)
    print(f"Wrote {len(records)} loss rows and {len(summaries)} epoch summaries.")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
