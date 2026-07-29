#!/usr/bin/env python3
"""
公平对比推理脚本（在 GPU 服务器上运行）

用法:
  python infer_fair_compare_server.py \
    --checkpoint-root data/experiments/r1_04_aug50/checkpoints \
    --output-dir data/experiments/r1_04_aug50/inference_fair \
    --ref-audio data/wavs_clean/001.wav \
    --ref-text "今天的天气很好，适合出去散步。" \
    --device cuda:0 \
    --epochs 4

生成公平版音频后，下载到本地写入对应位置：
  evaluation/data/generated/sft/

⚠️ 需要先设置环境变量:
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
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
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, nargs="+", default=[4])
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attn = "eager"  # 服务器无 flash-attn，统一用 eager

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
            wavs, sr = model.generate_voice_clone(
                text=text, language="Chinese",
                ref_audio=args.ref_audio, ref_text=args.ref_text)

            out = args.output_dir / f"epoch{ep}_{name}.wav"
            sf.write(str(out), wavs[0], sr)
            print(f"  {out.name} ({len(wavs[0])/sr:.1f}s)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
