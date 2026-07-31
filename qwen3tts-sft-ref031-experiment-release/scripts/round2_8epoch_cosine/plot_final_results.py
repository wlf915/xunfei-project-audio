import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_ROOT = Path(__file__).resolve().parents[1]
HP_ROOT = RUN_ROOT / "hparam_runs"

RUNS = [
    (
        "A: lr=5e-7",
        HP_ROOT / "A_lr5e-7_cosine8_ref031_seed42" / "train.log",
    ),
    (
        "B: lr=1e-6",
        HP_ROOT / "B_lr1e-6_cosine8_ref031_seed42" / "train.log",
    ),
    (
        "C: lr=1.5e-6",
        HP_ROOT / "C_lr1.5e-6_cosine8_ref031_seed42" / "train.log",
    ),
]

PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+"
    r"Loss:\s+([0-9.eE+-]+)\s+\|\s+LR:\s+([0-9.eE+-]+)"
)

STEPS_PER_EPOCH = 90
SMOOTH_WINDOW = 10

def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")

records = []

for label, log_path in RUNS:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    run_records = []

    for line in log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = PATTERN.search(line)
        if not match:
            continue

        epoch = int(match.group(1))
        step = int(match.group(2))
        loss = float(match.group(3))
        lr = float(match.group(4))
        global_micro_step = epoch * STEPS_PER_EPOCH + step
        epoch_position = epoch + step / STEPS_PER_EPOCH

        row = {
            "run": label,
            "epoch": epoch,
            "step": step,
            "global_micro_step": global_micro_step,
            "epoch_position": epoch_position,
            "loss": loss,
            "lr": lr,
        }

        records.append(row)
        run_records.append(row)

    if len(run_records) != 720:
        raise RuntimeError(
            f"{label}: expected 720 loss records, found {len(run_records)}"
        )

csv_path = RUN_ROOT / "loss_all_runs.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run",
            "epoch",
            "step",
            "global_micro_step",
            "epoch_position",
            "loss",
            "lr",
        ],
    )
    writer.writeheader()
    writer.writerows(records)

fig, ax = plt.subplots(figsize=(13, 7))

for label, _ in RUNS:
    rows = [row for row in records if row["run"] == label]
    xs = np.asarray([row["epoch_position"] for row in rows])
    losses = np.asarray([row["loss"] for row in rows])

    ax.plot(xs, losses, alpha=0.12, linewidth=0.7)

    smooth = moving_average(losses, SMOOTH_WINDOW)
    smooth_x = xs[SMOOTH_WINDOW - 1:]
    ax.plot(smooth_x, smooth, linewidth=2.2, label=label)

ax.set_title(
    "Full SFT Loss Comparison\n"
    "ref=031.wav, seed=42, 5% warmup + cosine"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss")
ax.set_xlim(0, 8)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(RUN_ROOT / "loss_comparison.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 6))

for label, _ in RUNS:
    rows = [row for row in records if row["run"] == label]
    xs = [row["epoch_position"] for row in rows]
    lrs = [row["lr"] for row in rows]
    ax.plot(xs, lrs, linewidth=2.0, label=label)

ax.set_title("Learning-rate Schedule: 5% Warmup + Cosine")
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning rate")
ax.set_xlim(0, 8)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(RUN_ROOT / "lr_schedule.png", dpi=200)
plt.close(fig)

for label, _ in RUNS:
    rows = [row for row in records if row["run"] == label]
    xs = np.asarray([row["epoch_position"] for row in rows])
    losses = np.asarray([row["loss"] for row in rows])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, losses, alpha=0.18, linewidth=0.8, label="raw loss")

    smooth = moving_average(losses, SMOOTH_WINDOW)
    smooth_x = xs[SMOOTH_WINDOW - 1:]
    ax.plot(
        smooth_x,
        smooth,
        linewidth=2.2,
        label=f"moving average ({SMOOTH_WINDOW})",
    )

    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_xlim(0, 8)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    safe_name = (
        label.replace(":", "")
        .replace("=", "")
        .replace(".", "_")
        .replace(" ", "_")
    )

    fig.savefig(RUN_ROOT / f"loss_{safe_name}.png", dpi=200)
    plt.close(fig)

print("Loss records:", len(records))
print("CSV:", csv_path)
print("Combined loss:", RUN_ROOT / "loss_comparison.png")
print("LR schedule:", RUN_ROOT / "lr_schedule.png")
