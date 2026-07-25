#!/usr/bin/env python3
"""
Phase 1: 音频预处理 —— 静音裁剪 + 响度归一化

- 静音裁剪：基于 RMS 能量阈值检测语音起止点，首尾各保留 50ms
- 响度归一化：RMS 归一化到 -20dBFS
- 输出到 data/wavs_clean/
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "wavs"
OUTPUT_DIR = BASE_DIR / "wavs_clean"
TEXT_FILE = BASE_DIR / "text.txt"
TARGET_RMS_DB = -20.0       # 目标 RMS 响度 (dBFS)
SILENCE_THRESHOLD_DB = -30  # 静音阈值（相对峰值 dB）
KEEP_MS = 50                # 首尾保留过渡 (ms)
MIN_DURATION_SEC = 1.5      # 裁剪后最小有效时长


@dataclass
class PreprocessResult:
    filename: str
    text_id: str
    orig_dur: float
    clean_dur: float
    orig_rms_db: float
    clean_rms_db: float
    orig_peak: float
    trim_lead_sec: float
    trim_trail_sec: float
    skipped: bool = False
    reason: str = ""
    output_path: Optional[Path] = None


def load_text(text_path: Path) -> dict[str, str]:
    """读取 text.txt，返回 {text_id: text_str}"""
    import re
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


def rms_db(y: np.ndarray) -> float:
    """计算信号 RMS 的 dBFS 值"""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def trim_silence(y: np.ndarray, sr: int) -> tuple[np.ndarray, float, float]:
    """
    基于能量阈值裁剪首尾静音。
    返回 (裁剪后信号, 裁掉的开头秒数, 裁掉的结尾秒数)
    """
    # 使用 librosa 的 trim，它比手动阈值更稳定
    yt, idx = librosa.effects.trim(
        y, top_db=-SILENCE_THRESHOLD_DB,
        frame_length=2048, hop_length=512
    )

    # 首尾各保留 KEEP_MS 过渡
    keep_samples = int(sr * KEEP_MS / 1000)
    start_sample = max(0, idx[0] - keep_samples)
    end_sample = min(len(y), idx[1] + keep_samples)
    yt = y[start_sample:end_sample]

    trim_lead = start_sample / sr
    trim_trail = (len(y) - end_sample) / sr
    return yt, trim_lead, trim_trail


def normalize_loudness(y: np.ndarray, sr: int, target_db: float = TARGET_RMS_DB) -> tuple[np.ndarray, float]:
    """RMS 响度归一化"""
    current_db = rms_db(y)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    yn = y * gain_linear
    # 防止削波
    peak = np.max(np.abs(yn))
    if peak > 0.98:
        yn = yn / peak * 0.95
    return yn.astype(np.float32), current_db


def process_one(wav_path: Path, output_path: Path) -> PreprocessResult:
    """处理单个 wav 文件"""
    y, sr = sf.read(str(wav_path))
    orig_dur = len(y) / sr
    orig_db = rms_db(y)
    orig_peak = float(np.max(np.abs(y)))

    # 1. 静音裁剪
    yt, lead, trail = trim_silence(y, sr)
    clean_dur = len(yt) / sr

    if clean_dur < MIN_DURATION_SEC:
        # 太短，跳过裁剪用原始音频
        yt = y.copy()
        lead, trail = 0.0, 0.0
        clean_dur = orig_dur

    # 2. 响度归一化
    yn, orig_db = normalize_loudness(yt, sr)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), yn, sr)

    clean_db = rms_db(yn)
    return PreprocessResult(
        filename=wav_path.name,
        text_id="",
        orig_dur=round(orig_dur, 3),
        clean_dur=round(clean_dur, 3),
        orig_rms_db=round(orig_db, 1),
        clean_rms_db=round(clean_db, 1),
        orig_peak=round(orig_peak, 3),
        trim_lead_sec=round(lead, 2),
        trim_trail_sec=round(trail, 2),
        output_path=output_path,
    )


def main():
    print("=" * 60)
    print("Phase 1: 音频预处理（静音裁剪 + 响度归一化）")
    print("=" * 60)

    text_map = load_text(TEXT_FILE)
    print(f"文本映射: {len(text_map)} 条")

    wav_files = sorted(INPUT_DIR.glob("*.wav"))
    if not wav_files:
        wav_files = sorted(INPUT_DIR.glob("*.WAV"))
    print(f"音频文件: {len(wav_files)} 个")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[PreprocessResult] = []
    skipped = 0

    for i, wav in enumerate(wav_files):
        stem = wav.stem  # "001", "002", ...
        output_path = OUTPUT_DIR / f"{stem}.wav"
        try:
            r = process_one(wav, output_path)
            # 关联 text_id：stem "001" → "000001"
            text_id = str(int(stem)).zfill(6)
            r.text_id = text_id
            results.append(r)

            status = (
                f"dur={r.orig_dur:.1f}→{r.clean_dur:.1f}s"
                f" | rms={r.orig_rms_db:.0f}→{r.clean_rms_db:.0f}dB"
                f" | trim={r.trim_lead_sec:.1f}+{r.trim_trail_sec:.1f}s"
            )
            print(f"  [{i+1:3d}/{len(wav_files)}] {wav.name} → {status}")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(wav_files)}] {wav.name} ✗ 错误: {e}")
            skipped += 1

    # 统计
    if results:
        clean_durs = [r.clean_dur for r in results]
        orig_durs = [r.orig_dur for r in results]
        lead_trims = [r.trim_lead_sec for r in results]
        trail_trims = [r.trim_trail_sec for r in results]
        orig_rms = [r.orig_rms_db for r in results]
        clean_rms = [r.clean_rms_db for r in results]

        print(f"\n{'=' * 60}")
        print(f"预处理完成统计:")
        print(f"  成功: {len(results)} / 跳过: {skipped}")
        print(f"  时长: {sum(orig_durs)/60:.1f} → {sum(clean_durs)/60:.1f} 分钟")
        print(f"  平均裁剪: 开头 {np.mean(lead_trims):.2f}s / 结尾 {np.mean(trail_trims):.2f}s")
        print(f"  RMS: {np.mean(orig_rms):.1f} ± {np.std(orig_rms):.1f} → "
              f"{np.mean(clean_rms):.1f} ± {np.std(clean_rms):.2f} dBFS")
        print(f"  输出目录: {OUTPUT_DIR}")
        print(f"  输出文件数: {len(list(OUTPUT_DIR.glob('*.wav')))}")

    # 保存处理报告
    import json
    report = {
        "target_rms_db": TARGET_RMS_DB,
        "silence_threshold_db": SILENCE_THRESHOLD_DB,
        "samples": [
            {
                "filename": r.filename,
                "text_id": r.text_id,
                "orig_dur": r.orig_dur,
                "clean_dur": r.clean_dur,
                "orig_rms_db": r.orig_rms_db,
                "clean_rms_db": r.clean_rms_db,
                "orig_peak": r.orig_peak,
                "trim_lead_sec": r.trim_lead_sec,
                "trim_trail_sec": r.trim_trail_sec,
            }
            for r in results
        ]
    }
    report_path = BASE_DIR / "preprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"  处理报告: {report_path}")


if __name__ == "__main__":
    main()
