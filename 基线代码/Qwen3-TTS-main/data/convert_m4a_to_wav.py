#!/usr/bin/env python3
"""
将 data/new/ 下的 100 条 .m4a 录音批量转换为 .wav 格式，并生成训练 JSONL 文件。

转换参数：
  - 采样率：24000 Hz（课题推荐 ≥16kHz，24kHz 为最佳）
  - 位深：16-bit PCM
  - 声道：单声道 mono
  - 输出目录：data/wavs/
  - JSONL 输出：data/train_raw.jsonl
"""

import os
import re
import subprocess
import json
from pathlib import Path

# 路径配置（基于脚本所在目录）
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "new"          # 原始 m4a 目录
OUTPUT_DIR = BASE_DIR / "wavs"        # 输出 wav 目录
TEXT_FILE = BASE_DIR / "text.txt"     # 文本标注文件
JSONL_OUTPUT = BASE_DIR / "train_raw.jsonl"  # 训练 JSONL

# 音频转换参数
TARGET_SR = 24000       # 24kHz
TARGET_BIT_DEPTH = 16   # 16-bit PCM
TARGET_CHANNELS = 1     # mono

# ref_audio：课题建议全数据集使用同一条参考音频以提升音色一致性
# 默认使用第一条音频作为参考音频
REF_AUDIO_INDEX = 1     # 使用 001.wav 作为 ref_audio


def parse_text_file(text_path: Path) -> dict:
    """
    解析 text.txt 文件，返回 {id: text} 的字典。
    文件格式：
        000001 今天的天气很好，适合出去散步。
        <空行>
        000002 请把窗户打开，让空气流通一下。
        ...
    """
    id_to_text = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 匹配 "000001 文本内容"
            match = re.match(r"^(\d+)\s+(.+)$", line)
            if match:
                audio_id = match.group(1)       # "000001"
                text = match.group(2)           # "今天的天气很好..."
                id_to_text[audio_id] = text
    return id_to_text


def convert_m4a_to_wav(input_path: Path, output_path: Path) -> bool:
    """
    使用 macOS 内置 afconvert 将 m4a 转为 wav。
    参数：24kHz, 16-bit PCM, mono
    """
    # afconvert 格式字符串：LEI16 = Little Endian 16-bit Integer
    # LEI16@24000 = 16-bit PCM at 24000 Hz
    cmd = [
        "afconvert",
        "-f", "WAVE",
        "-d", f"LEI{TARGET_BIT_DEPTH}@{TARGET_SR}",
        "-c", str(TARGET_CHANNELS),   # 0-based: 1 = mono actually... no, afconvert uses "1" for mono
        str(input_path),
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  [错误] afconvert 失败: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [错误] 转换超时: {input_path.name}")
        return False
    except Exception as e:
        print(f"  [错误] {e}")
        return False


def main():
    # 1. 解析文本文件
    print("=" * 60)
    print("[1/4] 解析文本文件...")
    id_to_text = parse_text_file(TEXT_FILE)
    print(f"  共解析 {len(id_to_text)} 条文本标注")

    # 2. 扫描 m4a 文件并建立映射关系
    # m4a 文件命名：001.m4a ~ 100.m4a
    # text ID 命名：000001 ~ 000100
    # 映射：001.m4a ↔ 000001 ↔ 001.wav
    print("\n[2/4] 扫描 m4a 音频文件...")
    m4a_files = sorted(INPUT_DIR.glob("*.m4a"))
    print(f"  共找到 {len(m4a_files)} 个 m4a 文件")

    # 建立 filename stem → text 的映射
    # 001 → text for 000001
    mapping = []
    for m4a_path in m4a_files:
        stem = m4a_path.stem     # "001"
        # 将 "001" 映射到 text ID "000001"
        text_id = str(int(stem)).zfill(6)  # "001" → 1 → "000001"
        text = id_to_text.get(text_id)
        if text is None:
            print(f"  [警告] 未找到 {stem} ({text_id}) 对应的文本，跳过")
            continue
        mapping.append((stem, text_id, text, m4a_path))

    mapping.sort(key=lambda x: x[0])
    print(f"  成功匹配 {len(mapping)} 条音频-文本对")

    # 3. 创建输出目录并批量转换
    print("\n[3/4] 开始转换 m4a → wav (24kHz, 16-bit, mono)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    ref_audio_rel_path = f"data/wavs/{str(REF_AUDIO_INDEX).zfill(3)}.wav"

    for idx, (stem, text_id, text, m4a_path) in enumerate(mapping):
        output_name = f"{stem}.wav"
        output_path = OUTPUT_DIR / output_name
        print(f"  [{idx+1:3d}/{len(mapping)}] {stem}.m4a → {output_name}  ", end="", flush=True)
        if convert_m4a_to_wav(m4a_path, output_path):
            # 验证输出文件
            size_kb = output_path.stat().st_size / 1024
            print(f"✓ ({size_kb:.0f} KB)")
            success_count += 1
        else:
            print("✗")

    print(f"\n  转换完成: {success_count}/{len(mapping)} 成功")

    # 4. 生成训练 JSONL
    print("\n[4/4] 生成训练 JSONL...")
    # 检查参考音频是否存在
    ref_wav = OUTPUT_DIR / f"{str(REF_AUDIO_INDEX).zfill(3)}.wav"
    if not ref_wav.exists():
        print(f"  [警告] 参考音频 {ref_wav.name} 不存在，请检查")

    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for stem, text_id, text, _ in mapping:
            record = {
                "audio": f"data/wavs/{stem}.wav",
                "text": text,
                "ref_audio": f"data/wavs/{str(REF_AUDIO_INDEX).zfill(3)}.wav",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  JSONL 已保存: {JSONL_OUTPUT}")
    print(f"  共 {len(mapping)} 条记录")
    print(f"  参考音频 (ref_audio): 统一使用 {str(REF_AUDIO_INDEX).zfill(3)}.wav")
    print()
    print("=" * 60)
    print("全部完成！输出文件：")
    print(f"  wav 目录: {OUTPUT_DIR}")
    print(f"  训练文件: {JSONL_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
