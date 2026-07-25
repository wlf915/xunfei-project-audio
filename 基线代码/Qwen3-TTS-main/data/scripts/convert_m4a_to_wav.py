#!/usr/bin/env python3
"""
将 .m4a 录音批量转换为 .wav 格式，并生成训练 JSONL。

用法:
  python convert_m4a_to_wav.py          # 默认 v1: data/new/ → data/wavs/
  python convert_m4a_to_wav.py v2       # v2: data/new_v2/ → data/wavs_v2/
  python convert_m4a_to_wav.py v1       # 显式 v1

转换参数：
  - 采样率：24000 Hz
  - 位深：16-bit PCM
  - 声道：单声道 mono
"""

import os
import re
import sys
import subprocess
import json
from pathlib import Path

# BASE_DIR = data/ （脚本在 data/scripts/ 下）
BASE_DIR = Path(__file__).resolve().parent.parent

# ── v1 / v2 配置 ──────────────────────────────────────────────────────────────
CONFIGS = {
    "v1": {
        "label":      "第一轮（自拟文本）",
        "input_dir":  BASE_DIR / "new",              # 原始 m4a
        "output_dir": BASE_DIR / "wavs",             # 输出 wav
        "text_file":  BASE_DIR / "metadata" / "text.txt",
        "jsonl_out":  BASE_DIR / "train_raw.jsonl",
        "ref_audio_index": 1,
    },
    "v2": {
        "label":      "第二轮（标贝语料）",
        "input_dir":  BASE_DIR / "new_v2",           # 原始 m4a
        "output_dir": BASE_DIR / "wavs_v2",          # 输出 wav
        "text_file":  BASE_DIR / "metadata" / "text_v2.txt",
        "jsonl_out":  BASE_DIR / "train_raw_v2.jsonl",
        "ref_audio_index": 1,
    },
}

# 音频转换参数
TARGET_SR = 24000        # 24kHz
TARGET_BIT_DEPTH = 16    # 16-bit PCM
TARGET_CHANNELS = 1      # mono

# ── 文本解析 ──────────────────────────────────────────────────────────────────
def parse_text_file(text_path: Path) -> dict[str, str]:
    """返回 {text_id(6位): text}"""
    id_to_text = {}
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                id_to_text[m.group(1)] = m.group(2)
    return id_to_text

# ── 音频转换 ──────────────────────────────────────────────────────────────────
def convert_m4a_to_wav(input_path: Path, output_path: Path) -> bool:
    """macOS afconvert: m4a → 24kHz 16-bit mono wav"""
    cmd = [
        "afconvert",
        "-f", "WAVE",
        "-d", f"LEI{TARGET_BIT_DEPTH}@{TARGET_SR}",
        "-c", str(TARGET_CHANNELS),
        str(input_path),
        str(output_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"  [错误] afconvert 失败: {r.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [错误] 转换超时: {input_path.name}")
        return False
    except Exception as e:
        print(f"  [错误] {e}")
        return False

# ── 主流程 ────────────────────────────────────────────────────────────────────
def run(version: str):
    cfg = CONFIGS[version]
    ref_idx = cfg["ref_audio_index"]

    print("=" * 60)
    print(f"m4a → wav 转换: {cfg['label']}")
    print("=" * 60)

    # ------ 1. 解析文本 ------
    print(f"\n[1/4] 解析文本: {cfg['text_file']}")
    id_to_text = parse_text_file(cfg["text_file"])
    print(f"      共 {len(id_to_text)} 条")

    # ------ 2. 扫描 m4a ------
    print(f"\n[2/4] 扫描 m4a: {cfg['input_dir']}")
    if not cfg["input_dir"].is_dir():
        print(f"      [错误] 目录不存在: {cfg['input_dir']}")
        print(f"      请将 {version} 录音 m4a 放入该目录后重试")
        return 1

    m4a_files = sorted(cfg["input_dir"].glob("*.m4a"))
    print(f"      找到 {len(m4a_files)} 个 m4a 文件")

    # 建立映射: stem("001") → text_id("000001") → text
    mapping = []
    for m4a_path in m4a_files:
        stem = m4a_path.stem
        text_id = str(int(stem)).zfill(6)
        text = id_to_text.get(text_id)
        if text is None:
            print(f"      [警告] {stem}.m4a (ID={text_id}) 无对应文本，跳过")
            continue
        mapping.append((stem, text_id, text, m4a_path))
    mapping.sort(key=lambda x: x[0])
    print(f"      成功匹配 {len(mapping)} 条")

    if not mapping:
        print("      无数据可处理，退出")
        return 1

    # ------ 3. 批量转换 ------
    print(f"\n[3/4] 转换 m4a → wav (24kHz, 16-bit, mono)")
    print(f"      输出: {cfg['output_dir']}")
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    ok = 0
    for i, (stem, _, _, m4a_path) in enumerate(mapping):
        out = cfg["output_dir"] / f"{stem}.wav"
        print(f"  [{i+1:3d}/{len(mapping)}] {m4a_path.name} → {out.name}  ", end="", flush=True)
        if convert_m4a_to_wav(m4a_path, out):
            kb = out.stat().st_size / 1024
            print(f"✓ ({kb:.0f} KB)")
            ok += 1
        else:
            print("✗")
    print(f"\n      完成: {ok}/{len(mapping)}")

    # ------ 4. 生成 JSONL ------
    print(f"\n[4/4] 生成训练 JSONL: {cfg['jsonl_out']}")

    ref_wav_name = f"{str(ref_idx).zfill(3)}.wav"
    ref_audio_rel = f"data/wavs_v2/{ref_wav_name}" if version == "v2" else f"data/wavs/{ref_wav_name}"

    ref_wav = cfg["output_dir"] / ref_wav_name
    if not ref_wav.exists():
        print(f"      [警告] 参考音频 {ref_wav_name} 不存在，请检查")

    with open(cfg["jsonl_out"], "w", encoding="utf-8") as f:
        for stem, _, text, _ in mapping:
            wav_rel = f"data/wavs_v2/{stem}.wav" if version == "v2" else f"data/wavs/{stem}.wav"
            record = {
                "audio": wav_rel,
                "text": text,
                "ref_audio": ref_audio_rel,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"      共 {len(mapping)} 条")
    print(f"      ref_audio: 统一取 {ref_audio_rel}")
    print()
    print("=" * 60)
    print(f"完成! wav → {cfg['output_dir']}")
    print(f"      jsonl → {cfg['jsonl_out']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if ver not in CONFIGS:
        print(f"未知版本: {ver}，可选: {list(CONFIGS.keys())}")
        sys.exit(1)
    raise SystemExit(run(ver))
