#!/usr/bin/env python3
"""
微调模型对比推理 —— 加载多个 checkpoint，用相同测试文本生成语音对比。

融合自同组同学 @audio_processing/qwen3_tts_finetuning/scripts/infer_exp01_compare.py
适配 MPS，支持多实验目录 + 多 checkpoint 批量推理。
"""

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# 测试文本（来自同学设计，覆盖 seen_style / unseen_short / unseen_long 三类）
DEFAULT_TEST_TEXTS = {
    "seen_style": "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long": "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="加载多个 SFT checkpoint，用固定测试文本生成对比语音。"
    )
    p.add_argument("--checkpoint-root", type=Path, required=True,
                   help="包含 checkpoint-epoch-N 子目录的路径")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="生成 WAV 的输出目录")
    p.add_argument("--speaker-name", default="speaker_test",
                   help="SFT 训练时的 speaker_name")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--device", default="mps",
                   help="推理设备 (cuda:0 / mps / cpu)")
    p.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="要加载的 epoch 列表，默认 0-4")
    p.add_argument("--texts", nargs="+",
                   help="自定义测试文本（不传则用默认三类文本）")
    return p.parse_args(argv)


def get_checkpoints(root: Path, epochs: list[int]) -> dict[str, Path]:
    """发现存在的 checkpoint 目录"""
    ckpts = {}
    for ep in epochs:
        ckpt_dir = root / f"checkpoint-epoch-{ep}"
        if ckpt_dir.exists() and (ckpt_dir / "config.json").exists():
            ckpts[f"epoch{ep}"] = ckpt_dir
    if not ckpts:
        raise FileNotFoundError(f"未在 {root} 找到任何 checkpoint-epoch-N 目录")
    return ckpts


def main():
    args = parse_args()

    # 确定测试文本
    if args.texts:
        test_texts = {f"custom_{i}": t for i, t in enumerate(args.texts)}
    else:
        test_texts = DEFAULT_TEST_TEXTS

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 发现 checkpoint
    checkpoints = get_checkpoints(args.checkpoint_root, args.epochs)

    print(f"设备: {args.device}")
    print(f"测试文本: {len(test_texts)} 条")
    print(f"Checkpoints: {list(checkpoints.keys())}")

    # MPS 不支持 flash_attention_2，自动降级
    attn = "sdpa" if args.device == "mps" else "flash_attention_2"

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n===== {ckpt_name} =====")

        model = Qwen3TTSModel.from_pretrained(
            str(ckpt_path),
            device_map=args.device,
            dtype=torch.bfloat16,
            attn_implementation=attn,
        )

        for text_name, text in test_texts.items():
            print(f"  生成: {text_name} → {text[:35]}...")

            wavs, sr = model.generate_custom_voice(
                text=text,
                language=args.language,
                speaker=args.speaker_name,
            )
            out_path = args.output_dir / f"{ckpt_name}_{text_name}.wav"
            sf.write(str(out_path), wavs[0], sr)
            dur = len(wavs[0]) / sr
            print(f"    保存: {out_path.name} ({dur:.1f}s, {sr}Hz)")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n全部完成: {len(checkpoints)} checkpoints × {len(test_texts)} texts "
          f"= {len(checkpoints) * len(test_texts)} 条语音")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
