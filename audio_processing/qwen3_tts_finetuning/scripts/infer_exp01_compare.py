import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


TEST_TEXTS = {
    "seen_style": "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long": (
        "人工智能语音合成技术正在快速发展，"
        "并逐渐应用到教育、医疗和智能交互等实际场景。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3-TTS checkpoints across fixed test texts."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help=(
            "Directory containing checkpoint-epoch-0, "
            "checkpoint-epoch-1 and checkpoint-epoch-2."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to save generated WAV files.",
    )
    parser.add_argument(
        "--speaker-name",
        default="student_voice",
        help="Speaker name used during fine-tuning.",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Generation language passed to Qwen3-TTS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        f"epoch{epoch}": args.checkpoint_root / f"checkpoint-epoch-{epoch}"
        for epoch in range(3)
    }

    missing = [
        str(path)
        for path in checkpoints.values()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "The following checkpoint directories do not exist:\n"
            + "\n".join(missing)
        )

    for checkpoint_name, checkpoint_path in checkpoints.items():
        print(f"\n===== Loading {checkpoint_name} =====")
        print(checkpoint_path)

        model = Qwen3TTSModel.from_pretrained(
            str(checkpoint_path),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        for text_name, text in TEST_TEXTS.items():
            print(f"Generating: {checkpoint_name} / {text_name}")

            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=args.language,
                speaker=args.speaker_name,
            )

            output_path = (
                args.output_dir
                / f"{checkpoint_name}_{text_name}.wav"
            )
            sf.write(output_path, wavs[0], sample_rate)

            print(f"Saved: {output_path}")
            print(f"Sample rate: {sample_rate}")

        del model
        torch.cuda.empty_cache()

    print("\nAll inference tasks completed.")


if __name__ == "__main__":
    main()
