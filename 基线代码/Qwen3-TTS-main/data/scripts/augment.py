#!/usr/bin/env python3
"""
Phase 3: 数据增强 —— 语速扰动 ±10% + 音高扰动 ±50Hz

对指定目录的 wav 文件做增强：
  - 语速 0.9x / 1.0x(原速) / 1.1x
  - 音高 -50Hz / 不变 / +50Hz
每个原始音频生成 2 条增强副本（不包含原始速度+原始音高那条）
输出到 data/wavs_augmented/
"""

import os
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "wavs_clean"   # Phase 1 预处理后的数据
OUTPUT_DIR = BASE_DIR / "wavs_augmented"
AUG_META = BASE_DIR / "reports" / "augment_meta.json"

SPEED_FACTORS = [0.9, 1.0, 1.1]      # 语速因子
PITCH_SHIFTS = [-50, 0, 50]           # 音高偏移（Hz → 半音转换）


def hz_to_semitones(hz: float, sr: int) -> float:
    """将 Hz 偏移转换为给定采样率下的半音步数"""
    if hz == 0:
        return 0.0
    # 1 semitone ≈ 6% frequency change
    # 50Hz shift at typical F0 ~150Hz ≈ 2 semitones
    return hz / 50 * 2  # rough: 50Hz ≈ 2 semitones


def augment_speed(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """语速扰动（通过重采样实现，不改变音高）"""
    return librosa.effects.time_stretch(y=y, rate=factor)


def augment_pitch(y: np.ndarray, sr: int, hz_shift: float) -> np.ndarray:
    """音高扰动"""
    n_steps = hz_to_semitones(hz_shift, sr)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def main():
    import sys

    # 支持传入特定的 ID 列表作为过滤
    target_ids = set()
    if len(sys.argv) > 1:
        # 从文件读取要增强的样本 ID
        id_file = Path(sys.argv[1])
        if id_file.exists():
            with open(id_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        target_ids.add(line)
            print(f"从 {id_file} 读取 {len(target_ids)} 个样本 ID 用于增强")

    print("=" * 60)
    print("Phase 3: 数据增强（语速 ±10% + 音高 ±50Hz）")
    print("=" * 60)

    wav_files = sorted(INPUT_DIR.glob("*.wav"))
    if target_ids:
        # 过滤：target_ids 里放的是 text_id "000001" 格式，
        # 需要映射到 wav 文件名 "001.wav"
        wav_files = [
            w for w in wav_files
            if str(int(w.stem)).zfill(6) in target_ids
        ]
        print(f"筛选后: {len(wav_files)} 个文件待增强")
    else:
        print(f"待增强: {len(wav_files)} 个文件")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = []
    total_gen = 0

    for i, wav in enumerate(wav_files):
        stem = wav.stem
        y, sr = sf.read(str(wav))

        # 对每种扰动组合生成
        variants = []
        for sp in SPEED_FACTORS:
            for ps in PITCH_SHIFTS:
                if sp == 1.0 and ps == 0:
                    continue  # 跳过原始版本

                # 生成增强音频
                y_aug = y.copy()
                if sp != 1.0:
                    y_aug = augment_speed(y_aug, sr, sp)
                if ps != 0:
                    y_aug = augment_pitch(y_aug, sr, ps)

                # 文件名
                sp_label = f"sp{int(sp*100):03d}"  # "sp090", "sp100", "sp110"
                ps_label = f"ps{ps:+03d}"          # "ps-50", "ps+00", "ps+50"
                out_name = f"{stem}_{sp_label}_{ps_label}.wav"
                out_path = OUTPUT_DIR / out_name

                sf.write(str(out_path), y_aug.astype(np.float32), sr)
                variants.append({
                    "filename": out_name,
                    "orig_stem": stem,
                    "speed": sp,
                    "pitch_hz": ps,
                    "duration_sec": round(len(y_aug) / sr, 3),
                })
                total_gen += 1

        meta.append({"orig_stem": stem, "augmented": variants})

        if (i + 1) % 20 == 0 or i == len(wav_files) - 1:
            print(f"  [{i+1:3d}/{len(wav_files)}] {stem}.wav → {len(variants)} variants  "
                  f"(累计 {total_gen} 条)")

    # 保存元数据
    AUG_META.parent.mkdir(parents=True, exist_ok=True)
    with open(AUG_META, "w", encoding="utf-8") as f:
        json.dump({
            "source_dir": str(INPUT_DIR),
            "speed_factors": SPEED_FACTORS,
            "pitch_shifts_hz": PITCH_SHIFTS,
            "total_original": len(wav_files),
            "total_augmented": total_gen,
            "entries": meta,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  增强完成: {len(wav_files)} × (3×3-1) = {total_gen} 条增强音频")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  元数据: {AUG_META}")


if __name__ == "__main__":
    main()
