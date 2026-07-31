#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="$(dirname "$SCRIPT_DIR")"
source "$RUN_ROOT/env.sh"

run_or_skip() {
    local exp_name="$1"
    local learning_rate="$2"
    local exp_dir="$HP_ROOT/$exp_name"

    if [ -f "$exp_dir/EXPERIMENT_COMPLETE" ]; then
        echo "Skipping completed experiment: $exp_name"
        return 0
    fi

    if [ -e "$exp_dir" ]; then
        echo "ERROR: incomplete experiment directory already exists:"
        echo "$exp_dir"
        echo "Refusing to overwrite it."
        exit 1
    fi

    bash "$SCRIPT_DIR/run_one_lr.sh" "$exp_name" "$learning_rate"
}

run_or_skip "A_lr5e-7_cosine8_ref031_seed42" "5e-7"
run_or_skip "B_lr1e-6_cosine8_ref031_seed42" "1e-6"
run_or_skip "C_lr1.5e-6_cosine8_ref031_seed42" "1.5e-6"

touch "$RUN_ROOT/ALL_EXPERIMENTS_COMPLETE"
echo "All three experiments completed successfully."
