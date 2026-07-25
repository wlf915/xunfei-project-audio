#!/usr/bin/env python3
"""
Phase 2: 数据质量自动检测 —— ASR 转写 + CER 打分 + A/B/C 分级

- 使用 FunASR Paraformer 逐条转写音频
- 计算 CER（字错率）：转写结果 vs 标注文本
- 综合打分：CER + 静音占比 + 音量特征
- 输出 A/B/C 分级和完整质量报告
"""

import os
import re
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent  # data/scripts/ → data/
INPUT_DIR = BASE_DIR / "wavs_clean"  # Phase 1 预处理后的数据
ORIGINAL_DIR = BASE_DIR / "wavs"      # 回退到原始数据
TEXT_FILE = BASE_DIR / "metadata" / "text.txt"
REPORT_OUT = BASE_DIR / "reports" / "quality_report_asr.json"

# 使用本地下载的模型（已通过 modelscope CLI 下载到本地）
ASR_MODEL_PATH = "/Users/wlf/.cache/modelscope/models/iic--speech_paraformer-vad-punc-zh"


@dataclass
class QualityResult:
    text_id: str          # "000001"
    filename: str         # "001.wav"
    ground_truth: str     # 标注文本
    asr_text: str         # ASR 转写文本
    cer: float            # 字错率
    duration_sec: float   # 音频时长
    peak_db: float        # 峰值
    rms_db: float         # RMS
    silence_ratio: float  # 静音占比（近似）
    grade: str = ""       # A/B/C
    issues: list[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


def load_text(text_path: Path) -> dict[str, str]:
    mapping = {}
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def char_error_rate(ref: str, hyp: str) -> float:
    """计算字错率 CER（Levenshtein 距离 / 参考字数）"""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    n, m = len(ref_chars), len(hyp_chars)

    if n == 0:
        return float(m)  # 全部算错

    # DP 计算编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost,  # substitution
            )

    return dp[n][m] / n


def text_normalize(text: str) -> str:
    """归一化文本以便 CER 比较"""
    # 去除标点符号和空格
    import string
    punc = string.punctuation + "，。！？、；：""''…—～《》（）【】「」"
    text = text.translate(str.maketrans("", "", punc))
    text = re.sub(r"\s+", "", text)
    return text


def estimate_silence_ratio(y: np.ndarray, sr: int, threshold_db: float = -40) -> float:
    """估算静音占比"""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return 1.0
    threshold = rms * (10 ** (threshold_db / 20))
    # 分帧
    frame_len = int(sr * 0.025)  # 25ms
    hop_len = int(sr * 0.010)    # 10ms
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return 0.0
    silent_frames = 0
    for i in range(n_frames):
        frame = y[i*hop_len : i*hop_len + frame_len]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < threshold:
            silent_frames += 1
    return silent_frames / n_frames


def assign_grade(result: QualityResult) -> QualityResult:
    """
    综合打分规则：
      A 级（优）：CER < 10%, 静音占比 < 15%
      B 级（良）：CER < 25%, 静音占比 < 30%
      C 级（差）：其余
    降级规则：
      CER > 35% → 直接 C
      静音占比 > 50% → 直接 C
      峰值过低 (< 0.05) → 降一级
    """
    cer = result.cer
    sil = result.silence_ratio

    if cer < 0.10 and sil < 0.15:
        grade = "A"
    elif cer < 0.25 and sil < 0.30:
        grade = "B"
    else:
        grade = "C"

    # 降级规则
    if cer > 0.35:
        grade = "C"
        result.issues.append(f"CER过高({cer:.0%})")
    if sil > 0.50:
        grade = "C"
        result.issues.append(f"静音占比过高({sil:.0%})")
    if result.peak_db < -35:
        # peak_db 已经是 dBFS，< -35 说明原始峰值 < ~0.018
        if grade == "A":
            grade = "B"
            result.issues.append("音量偏低")
        elif grade == "B":
            grade = "C"
            result.issues.append("音量极低")

    result.grade = grade
    return result


def build_asr_transcriber(device: str = "mps"):
    """加载 Paraformer ASR 模型（从本地缓存路径）"""
    from funasr import AutoModel
    model_path = ASR_MODEL_PATH
    print(f"  加载 ASR 模型: {model_path}")
    model = AutoModel(
        model=model_path,
        device=device,
        disable_pbar=True,
    )
    return model


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"

    print("=" * 60)
    print("Phase 2: 数据质量自动检测（ASR 转写 + CER 打分）")
    print("=" * 60)

    # 确定输入目录
    if INPUT_DIR.exists() and list(INPUT_DIR.glob("*.wav")):
        audio_dir = INPUT_DIR
        print(f"使用预处理数据: {audio_dir}")
    else:
        audio_dir = ORIGINAL_DIR
        print(f"预处理数据不存在，使用原始数据: {audio_dir}")

    text_map = load_text(TEXT_FILE)
    print(f"文本映射: {len(text_map)} 条")

    # 加载 ASR
    asr = build_asr_transcriber(device)
    print("  ASR 模型加载完成")

    # 建立音频文件映射: stem → wav_path
    wav_files = sorted(audio_dir.glob("*.wav"))
    stem_to_wav: dict[str, Path] = {}
    for wf in wav_files:
        stem_to_wav[wf.stem] = wf

    results: list[QualityResult] = []
    missing_count = 0
    asr_fail_count = 0

    for stem in sorted(stem_to_wav.keys(), key=lambda s: int(s)):
        wav_path = stem_to_wav[stem]
        text_id = str(int(stem)).zfill(6)
        ground_truth = text_map.get(text_id, "")

        if not ground_truth:
            print(f"  [{stem}] 无对应文本，跳过")
            missing_count += 1
            continue

        try:
            # 读取音频特征
            y, sr = sf.read(str(wav_path))
            dur = len(y) / sr
            peak = float(np.max(np.abs(y)))
            rms = np.sqrt(np.mean(y ** 2))
            peak_db = 20 * np.log10(peak + 1e-10)
            rms_db = 20 * np.log10(rms + 1e-10)
            sil_ratio = estimate_silence_ratio(y, sr)

            # ASR 转写
            asr_result = asr.generate(input=str(wav_path))
            if isinstance(asr_result, list) and asr_result:
                asr_text = asr_result[0].get("text", "")
            else:
                asr_text = ""

            # 计算 CER（归一化后比较）
            ref_norm = text_normalize(ground_truth)
            hyp_norm = text_normalize(asr_text)
            cer = char_error_rate(ref_norm, hyp_norm)

            r = QualityResult(
                text_id=text_id,
                filename=wav_path.name,
                ground_truth=ground_truth,
                asr_text=asr_text,
                cer=round(cer, 4),
                duration_sec=round(dur, 3),
                peak_db=round(peak_db, 1),
                rms_db=round(rms_db, 1),
                silence_ratio=round(sil_ratio, 3),
            )
            r = assign_grade(r)
            results.append(r)

            status = f"CER={r.cer:.1%} | grade={r.grade} | sil={r.silence_ratio:.0%}"
            print(f"  [{stem}] {status}")

        except Exception as e:
            print(f"  [{stem}] ASR 失败: {e}")
            asr_fail_count += 1
            # 创建一个默认 C 级结果
            r = QualityResult(
                text_id=text_id, filename=wav_path.name,
                ground_truth=ground_truth, asr_text="[ERROR]",
                cer=1.0, duration_sec=0, peak_db=-100, rms_db=-100,
                silence_ratio=1.0, grade="C",
                issues=[f"ASR异常: {str(e)[:80]}"],
            )
            results.append(r)

    # 统计
    if results:
        grades = {"A": 0, "B": 0, "C": 0}
        for r in results:
            grades[r.grade] = grades.get(r.grade, 0) + 1
        cers = [r.cer for r in results]

        print(f"\n{'=' * 60}")
        print(f"质检完成:")
        print(f"  总数: {len(results)}")
        print(f"  A 级 (优): {grades['A']} ({grades['A']/len(results)*100:.0f}%)")
        print(f"  B 级 (良): {grades['B']} ({grades['B']/len(results)*100:.0f}%)")
        print(f"  C 级 (差): {grades['C']} ({grades['C']/len(results)*100:.0f}%)")
        print(f"  CER 均值: {np.mean(cers):.3f} ({np.mean(cers)*100:.1f}%)")
        print(f"  CER 中位数: {np.median(cers):.3f} ({np.median(cers)*100:.1f}%)")
        print(f"  CER 最小/最大: {min(cers):.3f} / {max(cers):.3f}")
        print(f"  缺失文本: {missing_count}")
        print(f"  ASR 异常: {asr_fail_count}")

    # 保存质量报告
    report = {
        "asr_model": ASR_MODEL_PATH,
        "device": device,
        "audio_source": str(audio_dir),
        "total": len(results) + missing_count,
        "valid": len(results),
        "missing_text": missing_count,
        "asr_errors": asr_fail_count,
        "grade_distribution": {
            "A": grades.get("A", 0) if results else 0,
            "B": grades.get("B", 0) if results else 0,
            "C": grades.get("C", 0) if results else 0,
        },
        "cer_stats": {
            "mean": round(float(np.mean(cers)), 4) if results else None,
            "median": round(float(np.median(cers)), 4) if results else None,
            "min": round(float(min(cers)), 4) if results else None,
            "max": round(float(max(cers)), 4) if results else None,
        } if results else {},
        "samples": [
            {
                "text_id": r.text_id,
                "filename": r.filename,
                "ground_truth": r.ground_truth,
                "asr_text": r.asr_text,
                "cer": r.cer,
                "grade": r.grade,
                "duration_sec": r.duration_sec,
                "peak_db": r.peak_db,
                "rms_db": r.rms_db,
                "silence_ratio": r.silence_ratio,
                "issues": r.issues,
            }
            for r in sorted(results, key=lambda x: x.cer)
        ]
    }

    report_path = REPORT_OUT
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  质量报告: {report_path}")

    # 输出 CER 最高/最低的样本供人工验证
    print(f"\n  --- CER 最低的 5 条（最好的样本）---")
    for r in sorted(results, key=lambda x: x.cer)[:5]:
        print(f"  {r.filename}: CER={r.cer:.2%} | {r.ground_truth[:30]}...")

    print(f"\n  --- CER 最高的 5 条（疑似问题样本）---")
    for r in sorted(results, key=lambda x: x.cer, reverse=True)[:5]:
        print(f"  {r.filename}: CER={r.cer:.2%} | GT='{r.ground_truth[:30]}...' | ASR='{r.asr_text[:30]}...'")


if __name__ == "__main__":
    main()
