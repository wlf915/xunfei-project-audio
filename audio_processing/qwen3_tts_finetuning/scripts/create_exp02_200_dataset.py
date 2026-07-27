import argparse
import json
import shutil
from pathlib import Path


def parse_text_file(path: Path):
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if "|" in line:
                sid, text = line.split("|", 1)
            elif "\t" in line:
                sid, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Cannot parse line {line_no} in {path}: {line}"
                    )
                sid, text = parts

            items.append((int(sid.strip()), text.strip()))

    items.sort(key=lambda x: x[0])
    return items


def copy_wav(src_dir: Path, src_id: int, dst_dir: Path, dst_id: int):
    candidates = [
        src_dir / f"{src_id:03d}.wav",
        src_dir / f"{src_id:06d}.wav",
        src_dir / f"{src_id}.wav",
    ]

    src = None
    for c in candidates:
        if c.exists():
            src = c
            break

    if src is None:
        raise FileNotFoundError(
            f"Missing wav for source id {src_id} in {src_dir}"
        )

    dst = dst_dir / f"{dst_id:03d}.wav"
    shutil.copy2(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-wav-dir", type=Path, required=True)
    parser.add_argument("--old-text", type=Path, required=True)
    parser.add_argument("--new-wav-dir", type=Path, required=True)
    parser.add_argument("--new-text", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    args = parser.parse_args()

    old_items = parse_text_file(args.old_text)
    new_items = parse_text_file(args.new_text)

    print("old text items:", len(old_items))
    print("new text items:", len(new_items))

    if len(old_items) != 100:
        raise ValueError(f"Expected 100 old text items, got {len(old_items)}")
    if len(new_items) != 100:
        raise ValueError(f"Expected 100 new text items, got {len(new_items)}")
    if not args.ref_audio.exists():
        raise FileNotFoundError(args.ref_audio)

    wav_out = args.output_root / "wavs"
    meta_out = args.output_root / "metadata"
    proc_out = args.output_root / "processed"

    if args.output_root.exists():
        print(f"Output root exists: {args.output_root}")

    wav_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)
    proc_out.mkdir(parents=True, exist_ok=True)

    combined = []

    # old data: 001-100
    for src_id, text in old_items:
        dst_id = src_id
        dst_wav = copy_wav(args.old_wav_dir, src_id, wav_out, dst_id)
        combined.append((dst_id, dst_wav, text))

    # new data: 101-200
    for src_id, text in new_items:
        dst_id = src_id + 100
        dst_wav = copy_wav(args.new_wav_dir, src_id, wav_out, dst_id)
        combined.append((dst_id, dst_wav, text))

    combined.sort(key=lambda x: x[0])

    if len(combined) != 200:
        raise ValueError(f"Expected 200 combined samples, got {len(combined)}")

    text_200_path = meta_out / "text_200.txt"
    train_jsonl = meta_out / "train_raw.jsonl"
    test_jsonl = meta_out / "test_raw.jsonl"
    train_ids = meta_out / "train_ids.txt"
    test_ids = meta_out / "test_ids.txt"

    train_count = 0
    test_count = 0

    with (
        text_200_path.open("w", encoding="utf-8") as text_f,
        train_jsonl.open("w", encoding="utf-8") as train_f,
        test_jsonl.open("w", encoding="utf-8") as test_f,
        train_ids.open("w", encoding="utf-8") as train_id_f,
        test_ids.open("w", encoding="utf-8") as test_id_f,
    ):
        for sample_id, wav_path, text in combined:
            six_id = f"{sample_id:06d}"
            text_f.write(f"{six_id}\t{text}\n")

            item = {
                "audio": str(wav_path.resolve()),
                "text": text,
                "ref_audio": str(args.ref_audio.resolve()),
            }

            # 逢10留作测试集：010,020,...,200
            if sample_id % 10 == 0:
                test_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                test_id_f.write(six_id + "\n")
                test_count += 1
            else:
                train_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                train_id_f.write(six_id + "\n")
                train_count += 1

    print("combined samples:", len(combined))
    print("train samples:", train_count)
    print("test samples:", test_count)
    print("output root:", args.output_root)


if __name__ == "__main__":
    main()
