#!/usr/bin/env python3
"""
Phase 1: 音频预处理 —— 静音裁剪 + 响度归一化

用法:
  python preprocess.py       # 默认 v1
  python preprocess.py v2    # v2 数据
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import librosa

BASE_DIR = Path(__file__).resolve().parent.parent

VERSIONS = {
    "v1": {"input_dir": BASE_DIR / "wavs", "output_dir": BASE_DIR / "wavs_clean",
           "text_file": BASE_DIR / "metadata" / "text.txt", "label": "第一轮(自拟)"},
    "v2": {"input_dir": BASE_DIR / "wavs_v2", "output_dir": BASE_DIR / "wavs_v2_clean",
           "text_file": BASE_DIR / "metadata" / "text_v2.txt", "label": "第二轮(标贝)"},
}

TARGET_RMS_DB = -20.0
SILENCE_THRESHOLD_DB = -30
KEEP_MS = 50
MIN_DURATION_SEC = 1.5


@dataclass
class PreprocessResult:
    filename: str; text_id: str; orig_dur: float; clean_dur: float
    orig_rms_db: float; clean_rms_db: float; orig_peak: float
    trim_lead_sec: float; trim_trail_sec: float
    output_path: Path = None


def load_text(path: Path) -> dict[str, str]:
    import re
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            match = re.match(r"^(\d+)\s+(.+)$", line)
            if match: m[match.group(1)] = match.group(2)
    return m


def rms_db(y): return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)


def trim_silence(y, sr):
    yt, idx = librosa.effects.trim(y, top_db=-SILENCE_THRESHOLD_DB, frame_length=2048, hop_length=512)
    keep = int(sr * KEEP_MS / 1000)
    s, e = max(0, idx[0] - keep), min(len(y), idx[1] + keep)
    return y[s:e], s / sr, (len(y) - e) / sr


def normalize_loudness(y, sr, target=TARGET_RMS_DB):
    gain = 10 ** ((target - rms_db(y)) / 20)
    yn = y * gain
    if np.max(np.abs(yn)) > 0.98: yn = yn / np.max(np.abs(yn)) * 0.95
    return yn.astype(np.float32)


def process_one(wav, out):
    y, sr = sf.read(str(wav))
    orig_dur = len(y) / sr
    orig_peak = float(np.max(np.abs(y)))
    orig_db = rms_db(y)
    yt, lead, trail = trim_silence(y, sr)
    clean_dur = len(yt) / sr
    if clean_dur < MIN_DURATION_SEC: yt, lead, trail, clean_dur = y.copy(), 0.0, 0.0, orig_dur
    yn = normalize_loudness(yt, sr)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), yn, sr)
    return PreprocessResult(
        filename=wav.name, text_id="", orig_dur=round(orig_dur, 3),
        clean_dur=round(clean_dur, 3), orig_rms_db=round(orig_db, 1),
        clean_rms_db=round(rms_db(yn), 1), orig_peak=round(orig_peak, 3),
        trim_lead_sec=round(lead, 2), trim_trail_sec=round(trail, 2), output_path=out)


def run(ver):
    cfg = VERSIONS[ver]
    print("=" * 60)
    print(f"Phase 1: 音频预处理 — {cfg['label']}")
    print("=" * 60)

    text_map = load_text(cfg["text_file"])
    print(f"文本映射: {len(text_map)} 条")

    wavs = sorted(cfg["input_dir"].glob("*.wav")) or sorted(cfg["input_dir"].glob("*.WAV"))
    print(f"音频文件: {len(wavs)} 个")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    results = []

    for i, wav in enumerate(wavs):
        out = cfg["output_dir"] / f"{wav.stem}.wav"
        try:
            r = process_one(wav, out)
            r.text_id = str(int(wav.stem)).zfill(6)
            results.append(r)
            print(f"  [{i+1:3d}/{len(wavs)}] {wav.name} dur={r.orig_dur:.1f}→{r.clean_dur:.1f}s rms={r.orig_rms_db:.0f}→{r.clean_rms_db:.0f}dB trim={r.trim_lead_sec:.1f}+{r.trim_trail_sec:.1f}s")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(wavs)}] {wav.name} ✗ {e}")

    if results:
        ds = [r.clean_dur for r in results]; od = [r.orig_dur for r in results]
        ls = [r.trim_lead_sec for r in results]; ts = [r.trim_trail_sec for r in results]
        orm = [r.orig_rms_db for r in results]; crm = [r.clean_rms_db for r in results]
        print(f"\n预处理完成: {len(results)}/{len(wavs)}")
        print(f"  时长: {sum(od)/60:.1f}→{sum(ds)/60:.1f}min | 平均裁剪: 头{np.mean(ls):.2f}s 尾{np.mean(ts):.2f}s")
        print(f"  RMS: {np.mean(orm):.1f}±{np.std(orm):.1f} → {np.mean(crm):.1f}±{np.std(crm):.2f}dBFS")

    report_path = BASE_DIR / "reports" / f"preprocess_report_{ver}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"version": ver, "target_rms_db": TARGET_RMS_DB, "samples": [
            {"filename": r.filename, "text_id": r.text_id, "orig_dur": r.orig_dur, "clean_dur": r.clean_dur,
             "orig_rms_db": r.orig_rms_db, "clean_rms_db": r.clean_rms_db, "orig_peak": r.orig_peak,
             "trim_lead_sec": r.trim_lead_sec, "trim_trail_sec": r.trim_trail_sec} for r in results
        ]}, f, ensure_ascii=False, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"  报告: {report_path}")
    return 0


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if ver not in VERSIONS: print(f"未知版本: {ver}，可选: {list(VERSIONS)}"); sys.exit(1)
    raise SystemExit(run(ver))
