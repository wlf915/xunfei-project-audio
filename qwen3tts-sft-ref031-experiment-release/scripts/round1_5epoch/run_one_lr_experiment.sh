#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_SRC:?QWEN_SRC is not set}"
: "${MODEL_DIR:?MODEL_DIR is not set}"
: "${TRAIN_JSONL:?TRAIN_JSONL is not set}"
: "${TEST_JSONL:?TEST_JSONL is not set}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT is not set}"
: "${WORK_ROOT:?WORK_ROOT is not set}"
: "${HP_ROOT:?HP_ROOT is not set}"
: "${REF_AUDIO:?REF_AUDIO is not set}"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 EXPERIMENT_NAME LEARNING_RATE"
    exit 1
fi

EXP_NAME="$1"
LEARNING_RATE="$2"
EXP_DIR="$HP_ROOT/$EXP_NAME"

if [ -e "$EXP_DIR" ]; then
    echo "Experiment directory already exists: $EXP_DIR"
    exit 1
fi

mkdir -p "$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/results"

{
    echo "experiment_name=$EXP_NAME"
    echo "learning_rate=$LEARNING_RATE"
    echo "num_epochs=5"
    echo "batch_size=2"
    echo "gradient_accumulation_steps=4"
    echo "effective_batch_size=8"
    echo "seed=42"
    echo "speaker_name=student_voice"
    echo "ref_audio=$REF_AUDIO"
    echo "train_jsonl=$TRAIN_JSONL"
    echo "base_model=$MODEL_DIR"
} > "$EXP_DIR/config.txt"

sha256sum "$REF_AUDIO" > "$EXP_DIR/ref_audio.sha256"
sha256sum "$TRAIN_JSONL" > "$EXP_DIR/train_jsonl.sha256"

echo "===== Training $EXP_NAME ====="

cd "$QWEN_SRC/finetuning"

PYTHONHASHSEED=42 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
python -u "$TRAIN_SCRIPT" \
    --init_model_path "$MODEL_DIR" \
    --output_model_path "$EXP_DIR/checkpoints" \
    --train_jsonl "$TRAIN_JSONL" \
    --batch_size 2 \
    --lr "$LEARNING_RATE" \
    --num_epochs 5 \
    --speaker_name student_voice \
    > "$EXP_DIR/train.log" 2>&1

echo "===== Verifying checkpoints ====="

for epoch in 0 1 2 3 4; do
    checkpoint="$EXP_DIR/checkpoints/checkpoint-epoch-$epoch/model.safetensors"

    if [ ! -s "$checkpoint" ]; then
        echo "Missing checkpoint: $checkpoint"
        exit 1
    fi

    echo "checkpoint epoch $epoch OK"
done

echo "===== Starting 23-audio inference for every epoch ====="

EXP_DIR="$EXP_DIR" \
WORK_ROOT="$WORK_ROOT" \
TEST_JSONL="$TEST_JSONL" \
PATH="$PATH" \
bash "$WORK_ROOT/scripts/infer_all_epochs_23_and_cleanup.sh" \
    > "$EXP_DIR/infer_all_epochs_23.log" 2>&1

compare_count=$(find "$EXP_DIR/results/compare_audio" -maxdepth 1 -type f -name '*.wav' -size +0c | wc -l)
test_count=$(find "$EXP_DIR/results/test20" -type f -name 'test_*.wav' -size +0c | wc -l)

if [ "$compare_count" -ne 15 ]; then
    echo "Expected 15 compare files, found $compare_count"
    exit 1
fi

if [ "$test_count" -ne 100 ]; then
    echo "Expected 100 test files, found $test_count"
    exit 1
fi

remaining_checkpoints=$(find "$EXP_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l)

if [ "$remaining_checkpoints" -ne 0 ]; then
    echo "Some checkpoints were not removed"
    exit 1
fi

echo "Experiment completed successfully: $EXP_NAME"
echo "Compare audio: $compare_count"
echo "Test audio: $test_count"
