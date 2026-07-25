#!/usr/bin/env python3
"""
Phase 2: 数据质量快速检测（无需 ASR 模型下载）

基于音频信号特征打分：
  - 静音占比（能量阈值检测）
  - 音量特征（RMS 一致性、峰值削波检测）
  - 时长合理性
  - 频谱特征（高频能量占比、频谱质心稳定性）

输出 A/B/C 分级和完整质量报告，替代需要下载 ASR 模型的方案。
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "wavs_clean"   # Phase 1 预处理后
FALLBACK_DIR = BASE_DIR / "wavs"       # 回退
TEXT_FILE = BASE_DIR / "metadata" / "text.txt"


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


def compute_audio_features(y: np.ndarray, sr: int) -> dict:
    """计算音频的多维质量特征"""
    dur = len(y) / sr

    # 基本统计量
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak_db = 20 * np.log10(peak + 1e-10)
    rms_db = 20 * np.log10(rms + 1e-10)

    # 静音占比（基于帧能量）
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    frame_rms = np.array([
        np.sqrt(np.mean(y[i*hop_len:i*hop_len+frame_len] ** 2))
        for i in range(n_frames)
    ])
    silence_threshold = rms * 0.1  # -20dB relative
    silent_frames = np.sum(frame_rms < silence_threshold)
    silence_ratio = float(silent_frames / n_frames)

    # RMS 稳定性（帧间变化）
    frame_rms_db = 20 * np.log10(frame_rms + 1e-10)
    rms_std = float(np.std(frame_rms_db))

    # 削波检测
    clipping_ratio = float(np.sum(np.abs(y) > 0.98) / len(y))

    # 频谱质心（使用 FFT 近似）
    n_fft = min(2048, len(y))
    if n_fft >= 64:
        spec = np.abs(np.fft.rfft(y[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-10))
        # 高频能量占比 (>4kHz)
        high_mask = freqs > 4000
        high_energy_ratio = float(np.sum(spec[high_mask]) / (np.sum(spec) + 1e-10))
    else:
        centroid = 0.0
        high_energy_ratio = 0.0

    # 零交叉率（用于判断是否有语音内容）
    zcr = float(np.sum(np.abs(np.diff(np.signbit(y)))) / len(y))

    return {
        "duration_sec": round(dur, 3),
        "peak_db": round(peak_db, 1),
        "rms_db": round(rms_db, 1),
        "silence_ratio": round(silence_ratio, 3),
        "rms_std_db": round(rms_std, 1),
        "clipping_ratio": round(clipping_ratio, 5),
        "spectral_centroid_hz": round(centroid, 0),
        "high_freq_ratio": round(high_energy_ratio, 4),
        "zero_crossing_rate": round(zcr, 5),
    }


def compute_quality_score(feat: dict) -> tuple[float, list[str], str]:
    """
    综合质量评分（0-100），越高越好。

    评分维度：
      - 静音占比（30分）：< 10% → 满分，> 40% → 0分
      - RMS 稳定性（25分）：std < 5dB → 满分，> 15dB → 0分
      - 削波检测（20分）：无削波 → 满分
      - 时长合理性（15分）：3-6秒最佳
      - 音量（10分）：RMS 在合理范围
    """
    score = 0.0
    issues = []

    # 1. 静音占比（0-30分）
    sil = feat["silence_ratio"]
    if sil < 0.10:
        score += 30
    elif sil < 0.20:
        score += 25
    elif sil < 0.30:
        score += 15
        issues.append(f"静音占比较高({sil:.0%})")
    elif sil < 0.40:
        score += 8
        issues.append(f"静音占比偏高({sil:.0%})")
    else:
        score += 2
        issues.append(f"静音占比过高({sil:.0%})")

    # 2. RMS 稳定性（0-25分）
    rms_std = feat["rms_std_db"]
    if rms_std < 5:
        score += 25
    elif rms_std < 10:
        score += 18
    elif rms_std < 15:
        score += 10
        issues.append(f"音量波动较大(std={rms_std:.0f}dB)")
    else:
        score += 3
        issues.append(f"音量剧烈波动(std={rms_std:.0f}dB)")

    # 3. 削波（0-20分）
    clip = feat["clipping_ratio"]
    if clip < 0.0001:
        score += 20
    elif clip < 0.001:
        score += 10
        issues.append(f"轻微削波({clip:.2%})")
    else:
        score += 2
        issues.append(f"明显削波({clip:.2%})")

    # 4. 时长（0-15分）
    dur = feat["duration_sec"]
    if 3.0 <= dur <= 6.0:
        score += 15
    elif 2.0 <= dur <= 8.0:
        score += 10
    elif dur >= 1.5:
        score += 5
        issues.append(f"时长偏短({dur:.1f}s)")
    else:
        score += 1
        issues.append(f"时长过短({dur:.1f}s)")

    # 5. 音量（0-10分）
    rms_db = feat["rms_db"]
    if -22 <= rms_db <= -16:
        score += 10
    elif -26 <= rms_db <= -12:
        score += 6
    else:
        score += 2
        issues.append(f"音量异常(RMS={rms_db:.0f}dBFS)")

    # 特殊降级
    if feat["zero_crossing_rate"] < 0.001:
        score = min(score, 20)
        issues.append("零交叉率极低(可能为静音/纯音)")

    if feat["high_freq_ratio"] > 0.5:
        # 噪音可能
        score = max(0, score - 10)
        issues.append("高频能量占比过高(疑似噪音)")

    grade = "A" if score >= 80 else ("B" if score >= 55 else "C")
    return round(score, 1), issues, grade


def main():
    print("=" * 60)
    print("Phase 2: 数据质量快速检测（信号特征分析）")
    print("=" * 60)

    audio_dir = INPUT_DIR if INPUT_DIR.exists() and list(INPUT_DIR.glob("*.wav")) else FALLBACK_DIR
    print(f"音频目录: {audio_dir}")

    text_map = load_text(TEXT_FILE)
    print(f"文本映射: {len(text_map)} 条")

    wav_files = sorted(audio_dir.glob("*.wav"))
    stem_to_wav = {wf.stem: wf for wf in wav_files}

    results = []

    for stem in sorted(stem_to_wav.keys(), key=lambda s: int(s)):
        wav_path = stem_to_wav[stem]
        text_id = str(int(stem)).zfill(6)

        try:
            y, sr = sf.read(str(wav_path))
            feat = compute_audio_features(y, sr)
            score, issues, grade = compute_quality_score(feat)

            result = {
                "text_id": text_id,
                "filename": wav_path.name,
                "score": score,
                "grade": grade,
                "issues": issues,
                **feat,
            }
            results.append(result)
            print(f"  [{stem}] score={score:.0f} grade={grade} sil={feat['silence_ratio']:.0%} dur={feat['duration_sec']:.1f}s")
        except Exception as e:
            print(f"  [{stem}] 错误: {e}")
            results.append({
                "text_id": text_id,
                "filename": wav_path.name,
                "score": 0,
                "grade": "C",
                "issues": [f"读取失败: {str(e)[:80]}"],
                "error": True,
            })

    # 统计
    grades = {"A": 0, "B": 0, "C": 0}
    for r in results:
        grades[r["grade"]] = grades.get(r["grade"], 0) + 1
    scores = [r["score"] for r in results if not r.get("error")]

    print(f"\n{'=' * 60}")
    print(f"质检完成:")
    print(f"  总数: {len(results)}")
    print(f"  A 级 (≥80分): {grades['A']} ({grades['A']/len(results)*100:.0f}%)")
    print(f"  B 级 (55-79):  {grades['B']} ({grades['B']/len(results)*100:.0f}%)")
    print(f"  C 级 (<55):    {grades['C']} ({grades['C']/len(results)*100:.0f}%)")
    if scores:
        print(f"  平均分: {np.mean(scores):.1f}")
        print(f"  中位数: {np.median(scores):.1f}")
        print(f"  最低/最高: {min(scores):.0f}/{max(scores):.0f}")

    # 按分数排序输出
    top_10 = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
    bottom_10 = sorted(results, key=lambda x: x["score"])[:10]
    print(f"\n  --- Top-10 优质样本 ---")
    for r in top_10:
        print(f"  {r['filename']}: score={r['score']:.0f} grade={r['grade']}")
    print(f"\n  --- Bottom-10 问题样本 ---")
    for r in bottom_10:
        print(f"  {r['filename']}: score={r['score']:.0f} grade={r['grade']} issues={r['issues']}")

    # 保存报告
    report = {
        "method": "signal_feature_analysis",
        "audio_source": str(audio_dir),
        "total": len(results),
        "grade_distribution": {k: int(v) for k, v in grades.items()},
        "score_stats": {
            "mean": round(float(np.mean(scores)), 1) if scores else None,
            "median": round(float(np.median(scores)), 1) if scores else None,
            "min": round(float(min(scores)), 0) if scores else None,
            "max": round(float(max(scores)), 0) if scores else None,
        } if scores else {},
        "samples": sorted(results, key=lambda x: x["score"], reverse=True),
    }

    report_path = BASE_DIR / "reports" / "quality_report_fast.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\n  质量报告: {report_path}")


if __name__ == "__main__":
    main()
