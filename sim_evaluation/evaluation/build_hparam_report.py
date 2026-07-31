"""Build the fixed-enrollment hyperparameter SIM comparison chart."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_summary(summary_path: Path) -> list[dict[str, object]]:
    with Path(summary_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "learning_rate": row["learning_rate"],
                    "epoch": int(row["epoch"]),
                    "zero_shot_mean": float(row["zero_shot_mean"]),
                    "sft_mean": float(row["sft_mean"]),
                    "delta": float(row["delta"]),
                }
            )
    if len(rows) != 24:
        raise ValueError(f"Expected 24 summary rows, found {len(rows)}")
    return rows


def build_chart(summary_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = read_summary(summary_path)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["learning_rate"])].append(row)

    learning_rates = sorted(grouped, key=float)
    if len(learning_rates) != 3:
        raise ValueError(f"Expected 3 learning rates, found {len(learning_rates)}")

    colors = ("#1f77b4", "#d95f02", "#2b8c6b")
    markers = ("o", "s", "^")
    figure, axis = plt.subplots(figsize=(11.5, 6.8), dpi=160)
    for learning_rate, color, marker in zip(
        learning_rates, colors, markers, strict=True
    ):
        condition_rows = sorted(grouped[learning_rate], key=lambda row: int(row["epoch"]))
        epochs = [int(row["epoch"]) for row in condition_rows]
        if epochs != list(range(8)):
            raise ValueError(
                f"Learning rate {learning_rate} does not contain epochs 0 through 7"
            )
        axis.plot(
            epochs,
            [float(row["sft_mean"]) for row in condition_rows],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=f"SFT lr={learning_rate}",
        )

    zero_values = {float(row["zero_shot_mean"]) for row in rows}
    if len(zero_values) != 1:
        raise ValueError("The zero-shot baseline is not constant across conditions")
    zero_mean = zero_values.pop()
    axis.axhline(
        zero_mean,
        color="#555555",
        linestyle=(0, (6, 4)),
        linewidth=2,
        label=f"Zero-shot ref001 ({zero_mean:.4f})",
    )

    best = max(rows, key=lambda row: float(row["sft_mean"]))
    axis.annotate(
        (
            f"Best: lr={best['learning_rate']}, epoch {best['epoch']}\n"
            f"SIM {float(best['sft_mean']):.4f} "
            f"(Δ {float(best['delta']):+.4f})"
        ),
        xy=(int(best["epoch"]), float(best["sft_mean"])),
        xytext=(14, 14),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777777"},
        arrowprops={"arrowstyle": "->", "color": "#777777"},
    )

    axis.set_title(
        "Speaker Similarity by Learning Rate and Epoch",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=30,
    )
    figure.text(
        0.125,
        0.905,
        "Enrollment: 010 / 050 / 070  |  20 samples per condition  |  "
        "Zero-shot reference: 001",
        fontsize=10,
        color="#4a4a4a",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean speaker similarity (cosine)")
    axis.set_xticks(range(8))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower left", frameon=True, ncol=2)

    all_scores = [float(row["sft_mean"]) for row in rows] + [zero_mean]
    padding = 0.004
    axis.set_ylim(min(all_scores) - padding, max(all_scores) + padding)
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_chart(arguments.summary, arguments.output)
    print(arguments.output)
