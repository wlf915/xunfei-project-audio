import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

COMPARE_TEXTS = {
    "seen_style": "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long": "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def generate(model, text, speaker_name, seed):
    set_seed(seed)

    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        language="Chinese",
        speaker=speaker_name,
    )

    return wavs[0], sample_rate

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--test-jsonl", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--epoch-label", required=True)
parser.add_argument("--speaker-name", default="student_voice")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if not args.checkpoint.is_dir():
    raise FileNotFoundError(args.checkpoint)

if not args.test_jsonl.is_file():
    raise FileNotFoundError(args.test_jsonl)

compare_dir = args.output_root / "compare_audio"
test_dir = args.output_root / "test20" / args.epoch_label

compare_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading checkpoint: {args.checkpoint}", flush=True)

set_seed(args.seed)

model = Qwen3TTSModel.from_pretrained(
    str(args.checkpoint),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

print("===== Generating compare3 =====", flush=True)

for name, text in COMPARE_TEXTS.items():
    filename = f"{args.epoch_label}_{name}.wav"
    output = compare_dir / filename

    print(f"Generating: {filename}", flush=True)

    wav, sample_rate = generate(
        model=model,
        text=text,
        speaker_name=args.speaker_name,
        seed=args.seed,
    )

    sf.write(output, wav, sample_rate)

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Invalid compare output: {output}")

    print(f"Saved: {output}", flush=True)

print("===== Reading test20 =====", flush=True)

test_items = []

with args.test_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        audio_id = Path(item["audio"]).stem
        text = item["text"]
        test_items.append((audio_id, text))

if len(test_items) != 20:
    raise RuntimeError(f"Expected 20 test items, got {len(test_items)}")

manifest = test_dir / "metadata.tsv"

with manifest.open("w", encoding="utf-8") as mf:
    mf.write("audio_id\tfilename\ttext\n")

    print("===== Generating test20 =====", flush=True)

    for index, (audio_id, text) in enumerate(test_items, 1):
        filename = f"test_{audio_id}.wav"
        output = test_dir / filename

        print(f"[{index}/20] Generating {filename}: {text}", flush=True)

        wav, sample_rate = generate(
            model=model,
            text=text,
            speaker_name=args.speaker_name,
            seed=args.seed,
        )

        sf.write(output, wav, sample_rate)

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Invalid test output: {output}")

        mf.write(f"{audio_id}\t{filename}\t{text}\n")
        mf.flush()

        print(f"Saved: {output}", flush=True)

del model
torch.cuda.empty_cache()

print(f"Compare directory: {compare_dir}", flush=True)
print(f"Test directory: {test_dir}", flush=True)
print("Done.", flush=True)
