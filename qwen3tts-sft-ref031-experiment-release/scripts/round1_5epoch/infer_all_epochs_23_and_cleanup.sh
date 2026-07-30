#!/usr/bin/env bash
set -euo pipefail

: "${EXP_DIR:?EXP_DIR is not set}"
: "${WORK_ROOT:?WORK_ROOT is not set}"
: "${TEST_JSONL:?TEST_JSONL is not set}"

OUTPUT_ROOT="$EXP_DIR/results"

mkdir -p "$OUTPUT_ROOT/compare_audio"
mkdir -p "$OUTPUT_ROOT/test20"

for epoch in 0 1 2 3 4; do
    checkpoint="$EXP_DIR/checkpoints/checkpoint-epoch-$epoch"
    epoch_label="epoch$epoch"

    echo
    echo "=================================================="
    echo "Processing $epoch_label"
    echo "Checkpoint: $checkpoint"
    echo "=================================================="

    if [ ! -s "$checkpoint/model.safetensors" ]; then
        echo "Missing or invalid checkpoint: $checkpoint"
        exit 1
    fi

    PYTHONHASHSEED=42 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
    python -u "$WORK_ROOT/scripts/infer_one_epoch_compare3_test20.py" \
        --checkpoint "$checkpoint" \
        --test-jsonl "$TEST_JSONL" \
        --output-root "$OUTPUT_ROOT" \
        --epoch-label "$epoch_label" \
        --speaker-name student_voice \
        --seed 42

    compare_count=$(find "$OUTPUT_ROOT/compare_audio" -maxdepth 1 -type f -name "${epoch_label}_*.wav" -size +0c | wc -l)
    test_count=$(find "$OUTPUT_ROOT/test20/$epoch_label" -maxdepth 1 -type f -name 'test_*.wav' -size +0c | wc -l)
    metadata_lines=$(wc -l < "$OUTPUT_ROOT/test20/$epoch_label/metadata.tsv")

    echo "Compare audio count for $epoch_label: $compare_count"
    echo "Test audio count for $epoch_label: $test_count"
    echo "Metadata lines for $epoch_label: $metadata_lines"

    if [ "$compare_count" -ne 3 ]; then
        echo "Expected 3 compare files, found $compare_count"
        exit 1
    fi

    if [ "$test_count" -ne 20 ]; then
        echo "Expected 20 test files, found $test_count"
        exit 1
    fi

    if [ "$metadata_lines" -ne 21 ]; then
        echo "Expected 21 metadata lines, found $metadata_lines"
        exit 1
    fi

    case "$checkpoint" in
        "$EXP_DIR"/checkpoints/checkpoint-epoch-*)
            rm -rf -- "$checkpoint"
            echo "Deleted checkpoint after successful 23-audio inference: $checkpoint"
            ;;
        *)
            echo "Unsafe checkpoint path: $checkpoint"
            exit 1
            ;;
    esac

    echo "Completed $epoch_label"
done

echo
echo "=================================================="
echo "All epochs completed successfully."
echo "Expected compare audio: 15"
echo "Expected test audio: 100"
echo "All checkpoints have been removed."
echo "=================================================="
