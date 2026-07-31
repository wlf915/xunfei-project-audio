#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="$(dirname "$SCRIPT_DIR")"
source "$RUN_ROOT/env.sh"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 EXPERIMENT_NAME LEARNING_RATE"
    exit 1
fi

EXP_NAME="$1"
LEARNING_RATE="$2"
EXP_DIR="$HP_ROOT/$EXP_NAME"
TRAIN_SCRIPT="$RUN_ROOT/scripts/sft_12hz_seed42_cosine_infer_cleanup.py"

if [ -e "$EXP_DIR" ]; then
    echo "Experiment directory already exists: $EXP_DIR"
    exit 1
fi

mkdir -p "$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/results"

{
    echo "experiment_name=$EXP_NAME"
    echo "learning_rate=$LEARNING_RATE"
    echo "scheduler=cosine"
    echo "warmup_ratio=0.05"
    echo "num_epochs=8"
    echo "batch_size=2"
    echo "gradient_accumulation_steps=4"
    echo "effective_batch_size=8"
    echo "seed=42"
    echo "speaker_name=student_voice"
    echo "reference_audio=$REF_AUDIO"
    echo "train_jsonl=$TRAIN_JSONL"
    echo "test_jsonl=$TEST_JSONL"
    echo "base_model=$MODEL_DIR"
} > "$EXP_DIR/config.txt"

sha256sum "$REF_AUDIO" > "$EXP_DIR/ref_audio.sha256"
sha256sum "$TRAIN_JSONL" > "$EXP_DIR/train_jsonl.sha256"
sha256sum "$TEST_JSONL" > "$EXP_DIR/test_jsonl.sha256"

echo "===== Starting $EXP_NAME ====="

cd "$QWEN_SRC/finetuning"

env \
PYTHONHASHSEED=42 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$QWEN_SRC/finetuning${PYTHONPATH:+:$PYTHONPATH}" \
EPOCH_INFER_SCRIPT="$INFER_SCRIPT" \
EPOCH_TEST_JSONL="$TEST_JSONL" \
EPOCH_RESULT_ROOT="$EXP_DIR/results" \
python -u "$TRAIN_SCRIPT" \
    --init_model_path "$MODEL_DIR" \
    --output_model_path "$EXP_DIR/checkpoints" \
    --train_jsonl "$TRAIN_JSONL" \
    --batch_size 2 \
    --lr "$LEARNING_RATE" \
    --num_epochs 8 \
    --warmup_ratio 0.05 \
    --speaker_name student_voice \
    2>&1 | tee "$EXP_DIR/train.log"

all_count="$(find "$EXP_DIR/results" -type f -name '*.wav' | wc -l)"
compare_count="$(find "$EXP_DIR/results" -type f \( -name '*seen_style*.wav' -o -name '*unseen_short*.wav' -o -name '*unseen_long*.wav' \) | wc -l)"
test_count="$(find "$EXP_DIR/results" -type f -name 'test_*.wav' | wc -l)"
checkpoint_count="$(find "$EXP_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l)"

echo "Final verification for $EXP_NAME:"
echo "all wav: $all_count"
echo "compare wav: $compare_count"
echo "test wav: $test_count"
echo "remaining checkpoints: $checkpoint_count"

if [ "$all_count" -ne 184 ]; then
    echo "ERROR: expected 184 total wav files"
    exit 1
fi

if [ "$compare_count" -ne 24 ]; then
    echo "ERROR: expected 24 compare wav files"
    exit 1
fi

if [ "$test_count" -ne 160 ]; then
    echo "ERROR: expected 160 test wav files"
    exit 1
fi

if [ "$checkpoint_count" -ne 0 ]; then
    echo "ERROR: checkpoints remain"
    exit 1
fi

touch "$EXP_DIR/EXPERIMENT_COMPLETE"
echo "===== Completed $EXP_NAME ====="
