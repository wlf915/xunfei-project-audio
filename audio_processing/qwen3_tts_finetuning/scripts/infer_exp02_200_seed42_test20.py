import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-name", default="student_voice")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)

    if not args.test_jsonl.is_file():
        raise FileNotFoundError(args.test_jsonl)

    set_seed(args.seed)

    print("Loading checkpoint:")
    print(args.checkpoint)

    model = Qwen3TTSModel.from_pretrained(
        str(args.checkpoint),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    items = []
    with args.test_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            audio_id = Path(item["audio"]).stem
            text = item["text"]
            items.append((audio_id, text))

    print("test items:", len(items))

    manifest_path = args.output_dir / "metadata.tsv"

    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write("audio_id\tfilename\ttext\n")

        for audio_id, text in items:
            set_seed(args.seed)

            filename = f"test_{audio_id}.wav"
            output_path = args.output_dir / filename

            print(f"Generating {audio_id}: {text}")

            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker=args.speaker_name,
            )

            sf.write(output_path, wavs[0], sample_rate)

            mf.write(f"{audio_id}\t{filename}\t{text}\n")

            print(f"Saved: {output_path}")
            print(f"Sample rate: {sample_rate}")

    print("Manifest:", manifest_path)
    print("Done.")


if __name__ == "__main__":
    main()
