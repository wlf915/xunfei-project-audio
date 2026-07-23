import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/test JSONL files for Qwen3-TTS fine-tuning."
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Directory containing 001.wav to 100.wav.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="Text file containing sample ID and transcript.",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        required=True,
        help="Unified reference audio used by all samples.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Output path for training JSONL.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        required=True,
        help="Output path for test JSONL.",
    )
    return parser.parse_args()


def load_transcripts(text_file: Path) -> dict[int, str]:
    transcripts: dict[int, str] = {}

    with text_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid text format at line {line_number}: {line}"
                )

            sample_id_text, transcript = parts
            sample_id = int(sample_id_text)

            if sample_id in transcripts:
                raise ValueError(f"Duplicate sample ID: {sample_id_text}")

            transcripts[sample_id] = transcript

    return transcripts


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main() -> None:
    args = parse_args()

    if not args.wav_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {args.wav_dir}")
    if not args.text_file.is_file():
        raise FileNotFoundError(f"Text file not found: {args.text_file}")
    if not args.ref_audio.is_file():
        raise FileNotFoundError(
            f"Reference audio not found: {args.ref_audio}"
        )

    transcripts = load_transcripts(args.text_file)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for sample_id in sorted(transcripts):
        wav_path = args.wav_dir / f"{sample_id:03d}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Missing audio: {wav_path}")

        row = {
            "audio": str(wav_path.resolve()),
            "text": transcripts[sample_id],
            "ref_audio": str(args.ref_audio.resolve()),
        }

        if sample_id % 10 == 0:
            test_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"Train samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Train JSONL: {args.train_output}")
    print(f"Test JSONL: {args.test_output}")


if __name__ == "__main__":
    main()
