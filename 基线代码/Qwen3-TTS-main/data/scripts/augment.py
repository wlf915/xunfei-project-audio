#!/usr/bin/env python3
"""Phase 3: 数据增强。用法: python augment.py <ids_file> [v1|v2]"""

import json, sys
from pathlib import Path
import numpy as np; import soundfile as sf; import librosa

BASE_DIR = Path(__file__).resolve().parent.parent
SPEED_FACTORS = [0.9, 1.0, 1.1]
PITCH_SHIFTS = [-50, 0, 50]

VERSIONS = {
    "v1": {"input_dir": BASE_DIR / "wavs_clean", "output_dir": BASE_DIR / "wavs_augmented",
           "aug_meta": BASE_DIR / "reports" / "augment_meta_v1.json", "label": "第一轮(自拟)"},
    "v2": {"input_dir": BASE_DIR / "wavs_v2_clean", "output_dir": BASE_DIR / "wavs_v2_augmented",
           "aug_meta": BASE_DIR / "reports" / "augment_meta_v2.json", "label": "第二轮(标贝)"},
}


def hz_to_semitones(hz, sr): return hz / 50 * 2 if hz != 0 else 0.0


def augment_speed(y, sr, factor): return librosa.effects.time_stretch(y=y, rate=factor)


def augment_pitch(y, sr, hz): return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=hz_to_semitones(hz, sr))


def run(ids_file, ver):
    cfg = VERSIONS[ver]
    target_ids = set()
    if ids_file:
        with open(ids_file) as f:
            for line in f:
                line = line.strip()
                if line: target_ids.add(line)
    print(f"Phase 3: 数据增强 — {cfg['label']} | 目标IDs: {len(target_ids) if target_ids else '全部'}")

    wavs = sorted(cfg["input_dir"].glob("*.wav"))
    if target_ids:
        wavs = [w for w in wavs if str(int(w.stem)).zfill(6) in target_ids]
    print(f"待增强: {len(wavs)} 文件")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    meta, total = [], 0

    for i, wav in enumerate(wavs):
        y, sr = sf.read(str(wav)); variants = []
        for sp in SPEED_FACTORS:
            for ps in PITCH_SHIFTS:
                if sp == 1.0 and ps == 0: continue
                ya = y.copy()
                if sp != 1.0: ya = augment_speed(ya, sr, sp)
                if ps != 0: ya = augment_pitch(ya, sr, ps)
                out_name = f"{wav.stem}_sp{int(sp*100):03d}_ps{ps:+03d}.wav"
                sf.write(str(cfg["output_dir"] / out_name), ya.astype(np.float32), sr)
                variants.append({"filename": out_name, "orig_stem": wav.stem, "speed": sp, "pitch_hz": ps, "duration_sec": round(len(ya)/sr, 3)})
                total += 1
        meta.append({"orig_stem": wav.stem, "augmented": variants})
        if (i+1) % 20 == 0 or i == len(wavs)-1:
            print(f"  [{i+1:3d}/{len(wavs)}] {wav.stem} → {len(variants)} variants (累计{total}条)")

    cfg["aug_meta"].parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["aug_meta"], "w", encoding="utf-8") as f:
        json.dump({"version": ver, "source_dir": str(cfg["input_dir"]), "speed_factors": SPEED_FACTORS,
                    "pitch_shifts_hz": PITCH_SHIFTS, "total_original": len(wavs), "total_augmented": total,
                    "entries": meta}, f, ensure_ascii=False, indent=2)
    print(f"完成: {len(wavs)}→{total}条 → {cfg['output_dir']}\n  元数据: {cfg['aug_meta']}")
    return 0


if __name__ == "__main__":
    ids_file = sys.argv[1] if len(sys.argv) > 1 else None
    ver = sys.argv[2] if len(sys.argv) > 2 else "v1"
    if ver not in VERSIONS: print(f"未知版本: {ver}，可选: {list(VERSIONS)}"); sys.exit(1)
    raise SystemExit(run(ids_file, ver))
