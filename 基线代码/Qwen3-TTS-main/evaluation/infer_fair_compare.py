#!/usr/bin/env python3
"""
公平对比推理：用 generate_voice_clone 评测 SFT 模型

与 zero-shot 使用完全相同的推理 API，输入相同的 ref_audio + ref_text。
这样才能公平回答：SFT 是否让模型在使用同一参考音频时，音色克隆更好？

用法 (在 GPU 服务器):
  python evaluation/infer_fair_compare.py \
    --checkpoint-root data/experiments/r1_04_aug50/checkpoints_v3 \
    --output-dir data/experiments/r1_04_aug50/inference_fair \
    --ref-audio data/wavs_clean/001.wav \
    --ref-text "今天的天气很好，适合出去散步。" \
    --device cuda:0
"""

import argparse, os, sys
from pathlib import Path
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

TEST_TEXTS = {
    "seen_style":   "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long":  "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True,
                   help="包含 checkpoint-epoch-N 子目录")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ref-audio", required=True, help="参考音频路径")
    p.add_argument("--ref-text", required=True, help="参考音频文本")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, nargs="+", default=[0, 4, 9, 14])
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attn = "flash_attention_2" if "cuda" in args.device else "sdpa"

    for ep in args.epochs:
        ckpt_dir = args.checkpoint_root / f"checkpoint-epoch-{ep}"
        if not (ckpt_dir / "config.json").exists():
            print(f"  e{ep}: checkpoint 不存在，跳过")
            continue

        print(f"\n===== epoch {ep} =====")
        model = Qwen3TTSModel.from_pretrained(
            str(ckpt_dir), device_map=args.device,
            dtype=torch.bfloat16, attn_implementation=attn)

        for name, text in TEST_TEXTS.items():
            # ★ 关键改动: generate_voice_clone 代替 generate_custom_voice
            wavs, sr = model.generate_voice_clone(
                text=text, language="Chinese",
                ref_audio=args.ref_audio, ref_text=args.ref_text)

            out = args.output_dir / f"epoch{ep}_{name}.wav"
            sf.write(str(out), wavs[0], sr)
            print(f"  {out.name} ({len(wavs[0])/sr:.1f}s)")

        del model
        torch.cuda.empty_cache()

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
