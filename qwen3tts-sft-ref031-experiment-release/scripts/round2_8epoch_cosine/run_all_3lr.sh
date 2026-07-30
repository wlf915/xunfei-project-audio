#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="$(dirname "$SCRIPT_DIR")"

bash "$SCRIPT_DIR/run_one_lr.sh" "A_lr5e-7_cosine8_ref031_seed42" "5e-7"
bash "$SCRIPT_DIR/run_one_lr.sh" "B_lr1e-6_cosine8_ref031_seed42" "1e-6"
bash "$SCRIPT_DIR/run_one_lr.sh" "C_lr1.5e-6_cosine8_ref031_seed42" "1.5e-6"

touch "$RUN_ROOT/ALL_EXPERIMENTS_COMPLETE"
echo "All three experiments completed successfully."
