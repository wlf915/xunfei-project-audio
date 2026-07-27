import argparse
import os
import random
from pathlib import Path

import numpy as np
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


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-name", default="student_voice")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "epoch0": args.checkpoint_root / "checkpoint-epoch-0",
        "epoch1": args.checkpoint_root / "checkpoint-epoch-1",
        "epoch2": args.checkpoint_root / "checkpoint-epoch-2",
    }

    for name, path in checkpoints.items():
        if not path.is_dir():
            raise FileNotFoundError(path)

    for checkpoint_name, checkpoint_path in checkpoints.items():
        print(f"\n===== Loading {checkpoint_name} =====")
        print(checkpoint_path)

        set_seed(args.seed)

        model = Qwen3TTSModel.from_pretrained(
            str(checkpoint_path),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        for text_name, text in TEST_TEXTS.items():
            set_seed(args.seed)

            print(f"Generating: {checkpoint_name} / {text_name}")

            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker=args.speaker_name,
            )

            output_path = args.output_dir / f"{checkpoint_name}_{text_name}.wav"
            sf.write(output_path, wavs[0], sample_rate)

            print(f"Saved: {output_path}")
            print(f"Sample rate: {sample_rate}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
