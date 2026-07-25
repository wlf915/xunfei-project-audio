#!/usr/bin/env bash
# =============================================================================
# Qwen3-TTS 个性化语音合成 —— 全流程自动化 Pipeline
# =============================================================================
#
# 用法:
#   bash pipeline.sh all            # 运行全部（Mac 上只跑 Phase 1-4, 7-8）
#   bash pipeline.sh phase1         # 仅预处理
#   bash pipeline.sh phase2         # 仅质检
#   bash pipeline.sh phase3         # 仅数据增强
#   bash pipeline.sh phase4         # 仅构建实验分组
#   bash pipeline.sh phase5         # 提取 audio_codes（需 GPU / MPS）
#   bash pipeline.sh phase6 EXP     # SFT 训练（需 GPU），EXP 如 exp02_clean50_r1
#   bash pipeline.sh phase7 EXP     # 推理对比（需 GPU checkpoint）
#   bash pipeline.sh phase8         # SIM 评测
#   bash pipeline.sh quick-check    # 快速检查环境和数据
#
# 环境要求:
#   conda activate tts
#   export PIPELINE_DEVICE=mps      # 本地 Mac (默认)
#   export PIPELINE_DEVICE=cuda:0   # GPU 服务器
#
# =============================================================================

set -euo pipefail

# ── 路径配置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SCRIPTS_DIR="${DATA_DIR}/scripts"
EVAL_DIR="${SCRIPT_DIR}/evaluation"
FINETUNE_DIR="${SCRIPT_DIR}/finetuning"
EXPERIMENTS_DIR="${DATA_DIR}/experiments"
CONFIGS_DIR="${SCRIPT_DIR}/configs"

# ── 参数 ────────────────────────────────────────────────────────────────────
# ── Python 环境（自动检测 tts conda env）────────────────────────────────────
DEVICE="${PIPELINE_DEVICE:-mps}"

# 优先使用 conda tts 环境的 Python
if [ -f "/opt/homebrew/Caskroom/miniconda/base/envs/tts/bin/python3" ]; then
    PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/tts/bin/python3"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python3"
else
    PYTHON="$(which python3 2>/dev/null || echo python3)"
fi
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"

# 模型路径（可覆盖）
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"

# 训练超参（可覆盖）
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker_test}"

# 颜色
G='\033[1;32m' Y='\033[1;33m' R='\033[1;31m' N='\033[0m'
info()  { printf "${G}[INFO]${N}  %s\n" "$*"; }
warn()  { printf "${Y}[WARN]${N}  %s\n" "$*"; }
err()   { printf "${R}[ERR]${N}   %s\n" "$*"; }
title() { printf "\n${G}━━━ %s ━━━${N}\n" "$*"; }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: 音频预处理（静音裁剪 + 响度归一化）
# 输入: data/wavs/        输出: data/wavs_clean/
# ═══════════════════════════════════════════════════════════════════════════
phase1() {
    title "Phase 1: 音频预处理（静音裁剪 + 响度归一化）"

    if [ ! -d "${DATA_DIR}/wavs" ]; then
        err "找不到 data/wavs/，请先运行 convert_m4a_to_wav.py 转换原始录音"
        return 1
    fi

    ${PYTHON} "${SCRIPTS_DIR}/preprocess.py"
    info "预处理完成 → data/wavs_clean/"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: 数据质量检测（ASR 转写 + CER 打分）
# 输入: data/wavs_clean/  输出: data/reports/quality_report_asr.json
# ═══════════════════════════════════════════════════════════════════════════
phase2() {
    title "Phase 2: 数据质量检测"

    if [ -f "${DATA_DIR}/reports/quality_report_asr.json" ]; then
        warn "质检报告已存在，跳过（删掉报告文件可强制重跑）"
        return 0
    fi

    # 检查 ASR 模型是否已下载
    ASR_MODEL_DIR="${HOME}/.cache/modelscope/models/iic--speech_paraformer-vad-punc-zh"
    if [ ! -f "${ASR_MODEL_DIR}/model.pt" ]; then
        warn "ASR 模型未下载，尝试下载..."
        warn "如果下载失败，可手动运行: modelscope download --model iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir ${ASR_MODEL_DIR}"
        modelscope download \
            --model "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
            --local_dir "${ASR_MODEL_DIR}" 2>&1 || {
            warn "ASR 模型下载失败，使用快速质检方案 (quality_check_fast.py)"
            ${PYTHON} "${SCRIPTS_DIR}/quality_check_fast.py"
            return 0
        }
    fi

    ${PYTHON} "${SCRIPTS_DIR}/quality_check.py" "${DEVICE}"
    info "质检完成 → data/reports/quality_report_asr.json"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: 数据增强（语速 ±10% + 音高 ±50Hz）
# 输入: data/wavs_clean/  输出: data/wavs_augmented/
# ═══════════════════════════════════════════════════════════════════════════
phase3() {
    title "Phase 3: 数据增强"

    # 生成 top-50 ID 列表
    REPORT="${DATA_DIR}/reports/quality_report_asr.json"
    TOP50="${DATA_DIR}/metadata/top50_ids.txt"

    if [ -f "${REPORT}" ]; then
        ${PYTHON} -c "
import json
with open('${REPORT}') as f: r = json.load(f)
ab = [s for s in r['samples'] if s['grade'] in ('A','B')]
sort_k = 'cer' if 'cer' in ab[0] else 'score'
ab.sort(key=lambda x: x[sort_k], reverse=(sort_k=='score'))
with open('${TOP50}','w') as f:
    for s in ab[:50]: f.write(s['text_id']+'\n')
print(f'Top-50 IDs 已更新 ({len(ab[:50])} 条)')
" 2>&1
    else
        warn "无质检报告，使用全部 90 条训练 ID 做增强"
        ${PYTHON} -c "
with open('${TOP50}','w') as f:
    for i in range(1,101):
        if i % 10 != 0: f.write(f'{i:06d}\n')
"
    fi

    ${PYTHON} "${SCRIPTS_DIR}/augment.py" "${TOP50}"

    info "增强完成 → data/wavs_augmented/"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: 构建实验分组 JSONL
# 输入: data/wavs_clean/ + data/reports/  输出: data/experiments/exp*/
# ═══════════════════════════════════════════════════════════════════════════
phase4() {
    title "Phase 4: 构建 6 组实验 JSONL"

    ${PYTHON} "${SCRIPTS_DIR}/build_experiments.py"

    info "实验分组完成:"
    echo ""
    echo "  ┌────────────────────────┬──────────┬──────────┐"
    echo "  │ 实验组                 │ 训练条数 │ 测试条数 │"
    echo "  ├────────────────────────┼──────────┼──────────┤"
    for d in "${EXPERIMENTS_DIR}"/exp*/; do
        name=$(basename "$d")
        train=$(wc -l < "${d}/train_raw.jsonl" | tr -d ' ')
        test=$(wc -l < "${d}/test_raw.jsonl" | tr -d ' ')
        printf "  │ %-22s │ %8s │ %8s │\n" "$name" "$train" "$test"
    done
    echo "  └────────────────────────┴──────────┴──────────┘"
    echo ""
    info "JSONL 文件 → data/experiments/"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: 提取音频离散编码（audio_codes）
# 输入: JSONL          输出: *_with_codes.jsonl
# ═══════════════════════════════════════════════════════════════════════════
phase5() {
    title "Phase 5: 提取 audio_codes（需 GPU）"

    EXP="${1:-}"
    if [ -z "${EXP}" ]; then
        err "用法: bash pipeline.sh phase5 <实验名>"
        echo "  可用实验: $(ls "${EXPERIMENTS_DIR}" | grep '^exp' | tr '\n' ' ')"
        return 1
    fi

    EXP_DIR="${EXPERIMENTS_DIR}/${EXP}"
    INPUT="${EXP_DIR}/train_raw.jsonl"
    OUTPUT="${EXP_DIR}/train_with_codes.jsonl"

    if [ ! -f "${INPUT}" ]; then
        err "找不到 ${INPUT}"
        return 1
    fi

    ${PYTHON} "${FINETUNE_DIR}/prepare_data.py" \
        --device "${DEVICE}" \
        --tokenizer_model_path "${TOKENIZER_MODEL}" \
        --input_jsonl "${INPUT}" \
        --output_jsonl "${OUTPUT}"

    # 验证
    count=$(wc -l < "${OUTPUT}" | tr -d ' ')
    empty=$(grep -c '"audio_codes":\[\]' "${OUTPUT}" 2>/dev/null || echo 0)
    info "提取完成: ${count} 条, audio_codes 为空: ${empty}"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: SFT 微调训练
# 输入: train_with_codes.jsonl  输出: checkpoint-epoch-N/
# ═══════════════════════════════════════════════════════════════════════════
phase6() {
    title "Phase 6: SFT 微调训练（需 GPU）"

    EXP="${1:-}"
    if [ -z "${EXP}" ]; then
        err "用法: bash pipeline.sh phase6 <实验名>"
        echo "  可用实验: $(ls "${EXPERIMENTS_DIR}" | grep '^exp' | tr '\n' ' ')"
        return 1
    fi

    EXP_DIR="${EXPERIMENTS_DIR}/${EXP}"
    TRAIN_JSONL="${EXP_DIR}/train_with_codes.jsonl"
    MODEL_OUT="${EXP_DIR}/checkpoints"

    if [ ! -f "${TRAIN_JSONL}" ]; then
        err "找不到 ${TRAIN_JSONL}，请先运行 phase5"
        return 1
    fi

    echo ""
    echo "  实验:     ${EXP}"
    echo "  设备:     ${DEVICE}"
    echo "  Batch:    ${BATCH_SIZE}"
    echo "  累积步数: ${GRAD_ACCUM}"
    echo "  学习率:   ${LR}"
    echo "  Epochs:   ${NUM_EPOCHS}"
    echo "  说话人:   ${SPEAKER_NAME}"
    echo "  输出:     ${MODEL_OUT}"
    echo ""

    ${PYTHON} "${FINETUNE_DIR}/sft_12hz.py" \
        --init_model_path "${BASE_MODEL}" \
        --output_model_path "${MODEL_OUT}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --lr "${LR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --speaker_name "${SPEAKER_NAME}"

    info "训练完成 → ${MODEL_OUT}"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: 推理对比（多 checkpoint × 多测试文本）
# 输入: checkpoints/    输出: 对比 WAV 文件
# ═══════════════════════════════════════════════════════════════════════════
phase7() {
    title "Phase 7: 微调模型推理对比"

    EXP="${1:-}"
    if [ -z "${EXP}" ]; then
        err "用法: bash pipeline.sh phase7 <实验名>"
        echo "  可用实验: $(ls "${EXPERIMENTS_DIR}" | grep '^exp' | tr '\n' ' ')"
        return 1
    fi

    EXP_DIR="${EXPERIMENTS_DIR}/${EXP}"
    CKPT_DIR="${EXP_DIR}/checkpoints"
    OUT_DIR="${EXP_DIR}/inference_samples"

    if [ ! -d "${CKPT_DIR}" ]; then
        err "找不到 checkpoints: ${CKPT_DIR}"
        return 1
    fi

    ${PYTHON} "${EVAL_DIR}/infer_compare.py" \
        --checkpoint-root "${CKPT_DIR}" \
        --output-dir "${OUT_DIR}" \
        --speaker-name "${SPEAKER_NAME}" \
        --device "${DEVICE}" \
        --epochs 0 1 2 3 4

    info "推理对比完成 → ${OUT_DIR}"
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8: SIM 评测（zero-shot vs SFT 说话人相似度对比）
# 输入: evaluation/data/  输出: evaluation/results/
# ═══════════════════════════════════════════════════════════════════════════
phase8() {
    title "Phase 8: SIM 说话人相似度评测"

    ENROLL_DIR="${EVAL_DIR}/data/enrollment"
    GEN_DIR="${EVAL_DIR}/data/generated"
    OUT_DIR="${EVAL_DIR}/results"

    # Dry-run 检查
    echo "[1/2] 数据布局检查 ..."
    ${PYTHON} "${EVAL_DIR}/run_sim.py" \
        --enrollment-dir "${ENROLL_DIR}" \
        --generated-dir "${GEN_DIR}" \
        --output-dir "${OUT_DIR}" \
        --device "${DEVICE}" \
        --dry-run 2>&1 && echo "  dry-run 验证通过" || {
        warn "dry-run 失败，请按 README 放入 enrollment + generated 音频后重试"
        return 1
    }

    # 真实评测
    echo "[2/2] 运行 SIM 评测 ..."
    ${PYTHON} "${EVAL_DIR}/run_sim.py" \
        --enrollment-dir "${ENROLL_DIR}" \
        --generated-dir "${GEN_DIR}" \
        --output-dir "${OUT_DIR}" \
        --device "${DEVICE}"

    info "SIM 评测完成 → ${OUT_DIR}"
}


# ═══════════════════════════════════════════════════════════════════════════
# 快速环境检查
# ═══════════════════════════════════════════════════════════════════════════
quick_check() {
    title "环境与数据快速检查"

    echo ""
    echo "  Python:   $(${PYTHON} --version 2>&1)"
    echo "  设备:     ${DEVICE}"
    echo ""

    # PyTorch
    ${PYTHON} -c "
import torch; print(f'  PyTorch:  {torch.__version__}')
print(f'  MPS:      {torch.backends.mps.is_available()}')
" 2>&1

    # 数据统计
    echo ""
    echo "  数据文件:"
    for d in wavs wavs_clean wavs_augmented new; do
        dpath="${DATA_DIR}/${d}"
        cnt=0
        if [ -d "$dpath" ]; then
            cnt=$(find "$dpath" -type f \( -name "*.wav" -o -name "*.m4a" \) 2>/dev/null | wc -l | tr -d ' ')
        fi
        sz=$(du -sh "$dpath" 2>/dev/null | cut -f1)
        echo "    ${d}/  ${cnt} files, ${sz:-0}"
    done

    # JSONL
    echo ""
    echo "  实验 JSONL:"
    for d in "${EXPERIMENTS_DIR}"/exp*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        echo "    ${name}: $(wc -l < "${d}/train_raw.jsonl" 2>/dev/null | tr -d ' ') train + $(wc -l < "${d}/test_raw.jsonl" 2>/dev/null | tr -d ' ') test"
    done

    # 质检
    if [ -f "${DATA_DIR}/reports/quality_report_asr.json" ]; then
        echo ""
        echo "  质检报告: $(du -sh "${DATA_DIR}/reports/quality_report_asr.json" | cut -f1)"
        ${PYTHON} -c "
import json
with open('${DATA_DIR}/reports/quality_report_asr.json') as f:
    r = json.load(f)
print(f'    A/B/C: {r[\"grade_distribution\"]}')
print(f'    CER mean: {r.get(\"cer_stats\",{}).get(\"mean\",\"N/A\")}')
" 2>&1
    fi

    echo ""
    info "检查完成。运行 bash pipeline.sh all 开始全流程。"
}


# ═══════════════════════════════════════════════════════════════════════════
# 全量运行（Mac 本地可跑的步骤: 1-4, 7-8）
# GPU 步骤 (5-6) 需要手动指定实验名在 GPU 机器上运行
# ═══════════════════════════════════════════════════════════════════════════
run_all_mac() {
    title "全流程 Pipeline（Mac 本地阶段）"
    echo ""
    echo "  Phase 1-4:  数据处理（Mac 本地）"
    echo "  Phase 5-6:  需 GPU，请在学校服务器上运行"
    echo "  Phase 7-8:  评测推理（Mac 本地）"
    echo ""

    phase1
    phase2
    phase3
    phase4

    echo ""
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "  Phase 1-4 完成！"
    info ""
    info "  下一步（GPU 服务器）:"
    for exp in exp00_baseline_r1 exp01_clean30_r1 exp02_clean50_r1 exp03_clean80_r1 exp04_aug50_r1 exp05_full_clean; do
        echo "    bash pipeline.sh phase5 ${exp}"
        echo "    bash pipeline.sh phase6 ${exp}"
    done
    info ""
    info "  训练后回到 Mac 运行评测:"
    echo "    bash pipeline.sh phase7 exp02_clean50_r1"
    echo "    bash pipeline.sh phase8"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}


# ═══════════════════════════════════════════════════════════════════════════
# 调度入口
# ═══════════════════════════════════════════════════════════════════════════
case "${1:-help}" in
    all)         run_all_mac ;;
    phase1)      phase1 ;;
    phase2)      phase2 ;;
    phase3)      phase3 ;;
    phase4)      phase4 ;;
    phase5)      phase5 "${2:-}" ;;
    phase6)      phase6 "${2:-}" ;;
    phase7)      phase7 "${2:-}" ;;
    phase8)      phase8 ;;
    quick-check) quick_check ;;
    help|--help|-h)
        sed -n '2,30p' "$0"
        ;;
    *)
        echo "未知命令: $1"
        echo "用法: bash pipeline.sh [all|phase1-8|quick-check|help]"
        ;;
esac
