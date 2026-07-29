#!/usr/bin/env python3
"""生成 zero-shot 基线 → evaluation/data/generated/zero_shot/"""
import torch, soundfile as sf
from pathlib import Path
from qwen_tts import Qwen3TTSModel

OUT = Path(__file__).resolve().parent / "data" / "generated" / "zero_shot"
OUT.mkdir(parents=True, exist_ok=True)

# 与 Phase 7 相同的 3 条测试文本
TESTS = {
    "seen_style":   "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long":  "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}

ref_audio = str(Path(__file__).resolve().parent.parent / "data" / "wavs_clean" / "001.wav")
ref_text  = "今天的天气很好，适合出去散步。"

print("加载 Base 模型...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="mps", dtype=torch.bfloat16, attn_implementation="sdpa")

for name, text in TESTS.items():
    out = OUT / f"{name}.wav"
    if out.exists():
        print(f"  {name}.wav 已存在，跳过")
        continue
    wavs, sr = model.generate_voice_clone(text=text, language="Chinese",
        ref_audio=ref_audio, ref_text=ref_text)
    sf.write(str(out), wavs[0], sr)
    print(f"  {name}.wav done ({len(wavs[0])/sr:.1f}s)")

print(f"\nZero-shot 基线 → {OUT}")
