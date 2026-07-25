#!/usr/bin/env python3
"""
Phase 4: 构建实验分组 —— 基于质检结果生成多组训练 JSONL

6 组消融实验（R1 = 第一轮自拟数据）：
  exp00_baseline_r1  原始 100 条，未清洗，训练集 90 / 测试集 10
  exp01_clean30_r1   预处理后 A/B 级 Top-30，训练 30
  exp02_clean50_r1   预处理后 A/B 级 Top-50，训练 50
  exp03_clean80_r1   预处理后过滤 C 级，训练 ~80
  exp04_aug50_r1     clean50 + 语速/音高增强，训练 ~50+增强
  exp05_full_clean   clean80（最大规模清洗后全量），训练 ~80

统一变量：
  - test_set: 编号能被 10 整除的 10 条（固定不变）
  - ref_audio: 统一使用质量最高的 A 级样本作为 ref_audio
  - audio 路径相对于 data/ 目录
"""

import os
import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
QUALITY_REPORT = BASE_DIR / "reports" / "quality_report_asr.json"
TEXT_FILE = BASE_DIR / "metadata" / "text.txt"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# 路径约定
WAVS_ORIGINAL = "data/wavs"       # 相对于项目根目录
WAVS_CLEAN = "data/wavs_clean"
WAVS_AUG = "data/wavs_augmented"


def load_text(text_path: Path) -> dict[str, str]:
    mapping = {}
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                mapping[m.group(1)] = m.group(2)
    return mapping


def is_test_sample(text_id: str) -> bool:
    """编号能被 10 整除的作为测试集"""
    num = int(text_id)
    return num % 10 == 0


def load_quality_report() -> dict:
    """加载质检报告，返回 {text_id → sample_info} 的索引"""
    if not QUALITY_REPORT.exists():
        return {}
    with open(QUALITY_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    index = {}
    for s in report.get("samples", []):
        index[s["text_id"]] = s
    return index


def select_ref_audio(quality_index: dict, text_map: dict) -> tuple[str, str]:
    """从 A 级样本中选择最佳参考音频：CER 最低 + 时长适中"""
    a_samples = [
        (tid, q) for tid, q in quality_index.items()
        if q["grade"] == "A" and 3.0 <= q["duration_sec"] <= 6.0
    ]
    if not a_samples:
        # 降级到 B 级
        a_samples = [
            (tid, q) for tid, q in quality_index.items()
            if q["grade"] in ("A", "B") and 2.5 <= q["duration_sec"] <= 7.0
        ]
    if not a_samples:
        # 随便选一个
        a_samples = [(tid, q) for tid, q in quality_index.items()]

    # 按质量分数排序，选最好的（兼容 cer 和 score 两种报告格式）
    sort_key = "cer" if "cer" in a_samples[0][1] else "score"
    a_samples.sort(key=lambda x: x[1][sort_key], reverse=(sort_key == "score"))
    best_tid = a_samples[0][0]
    best_stem = str(int(best_tid)).zfill(3)
    ref_path = f"{WAVS_CLEAN}/{best_stem}.wav"
    ref_text = text_map.get(best_tid, "")
    return ref_path, ref_text


def build_jsonl(
    exp_name: str,
    sample_ids: list[str],
    text_map: dict[str, str],
    wav_dir: str,
    ref_audio: str,
    output_path: Path,
):
    """为指定样本 ID 列表生成训练 JSONL"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text_id in sorted(sample_ids, key=lambda x: int(x)):
            stem = str(int(text_id)).zfill(3)
            text = text_map.get(text_id, "")
            record = {
                "audio": f"{wav_dir}/{stem}.wav",
                "text": text,
                "ref_audio": ref_audio,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def build_jsonl_with_aug(
    exp_name: str,
    clean_ids: list[str],
    text_map: dict[str, str],
    ref_audio: str,
    aug_meta_path: Path,
    output_path: Path,
):
    """构建含增强样本的训练 JSONL（原始 clean + 增强变体）"""
    # 先加原始 clean 样本
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取增强元数据
    aug_map = {}  # stem → list of variant entries
    if aug_meta_path.exists():
        with open(aug_meta_path, encoding="utf-8") as f:
            aug_meta = json.load(f)
        for entry in aug_meta["entries"]:
            aug_map[entry["orig_stem"]] = entry["augmented"]

    records = []
    for text_id in sorted(clean_ids, key=lambda x: int(x)):
        stem = str(int(text_id)).zfill(3)
        text = text_map.get(text_id, "")

        # 原始 clean 版本
        records.append({
            "audio": f"{WAVS_CLEAN}/{stem}.wav",
            "text": text,
            "ref_audio": ref_audio,
        })

        # 增强版本
        if stem in aug_map:
            for variant in aug_map[stem]:
                records.append({
                    "audio": f"{WAVS_AUG}/{variant['filename']}",
                    "text": text,
                    "ref_audio": ref_audio,
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return output_path, len(records)


def main():
    print("=" * 60)
    print("Phase 4: 构建实验分组 JSONL")
    print("=" * 60)

    text_map = load_text(TEXT_FILE)
    print(f"文本映射: {len(text_map)} 条")

    quality_index = load_quality_report()
    has_quality = len(quality_index) > 0
    print(f"质检数据: {'可用' if has_quality else '不可用（将按默认规则分组）'}")

    # ── 选择参考音频 ────────────────────────────────────────
    if has_quality:
        ref_audio, ref_text = select_ref_audio(quality_index, text_map)
    else:
        # 默认用 001.wav
        ref_audio = f"{WAVS_CLEAN}/001.wav"
        ref_text = text_map.get("000001", "")

    ref_stem = Path(ref_audio).stem
    print(f"参考音频 (ref_audio): {ref_audio}")
    print(f"参考文本: {ref_text[:40]}...")

    # ── 划分训练/测试 ───────────────────────────────────────
    all_ids = []
    # text.txt 里的 ID 是六位 "000001" 格式
    for tid in text_map:
        all_ids.append(tid)
    all_ids.sort()

    train_ids_all = [tid for tid in all_ids if not is_test_sample(tid)]
    test_ids = [tid for tid in all_ids if is_test_sample(tid)]

    print(f"全量样本: {len(all_ids)} (训练 {len(train_ids_all)} + 测试 {len(test_ids)})")
    print(f"测试集 IDs: {', '.join(test_ids)}")

    # ── 按质量排序（用于筛选） ────────────────────────────────
    if has_quality:
        # 确定排序字段（兼容 cer 和 score 两种报告格式）
        sample0 = next(iter(quality_index.values()))
        sort_field = "cer" if "cer" in sample0 else "score"
        sort_reverse = (sort_field == "score")  # score 越高越好，cer 越低越好

        train_quality = [
            (tid, quality_index[tid])
            for tid in train_ids_all
            if tid in quality_index
        ]
        train_quality.sort(key=lambda x: x[1][sort_field], reverse=sort_reverse)

        # 过滤 C 级
        train_ab = [(tid, q) for tid, q in train_quality if q["grade"] in ("A", "B")]
        train_clean_ids = [tid for tid, _ in train_ab]
        train_sorted_ids = [tid for tid, _ in train_quality]  # 包括 C 级
    else:
        train_clean_ids = train_ids_all
        train_sorted_ids = train_ids_all

    N_clean = len(train_clean_ids)
    print(f"训练集 A/B 级（可入选实验）: {N_clean}")
    print(f"  A 级: {sum(1 for tid in train_clean_ids if quality_index.get(tid, {}).get('grade') == 'A') if has_quality else 'N/A'}")
    print(f"  B 级: {sum(1 for tid in train_clean_ids if quality_index.get(tid, {}).get('grade') == 'B') if has_quality else 'N/A'}")

    # ── 构建 6 组实验 ────────────────────────────────────────
    experiments = {}

    # exp00: Baseline-R1（原始未清洗数据）
    experiments["exp00_baseline_r1"] = {
        "description": "Baseline-R1: 原始100条未清洗，训练90/测试10",
        "train_ids": train_ids_all,
        "test_ids": test_ids,
        "wav_dir": WAVS_ORIGINAL,
        "use_aug": False,
    }

    # exp01: Clean-30（Top-30 A/B 级）
    top30 = train_clean_ids[:30] if len(train_clean_ids) >= 30 else train_clean_ids
    experiments["exp01_clean30_r1"] = {
        "description": "Clean-30: 预处理后质量最优30条（A/B级）",
        "train_ids": top30,
        "test_ids": test_ids,
        "wav_dir": WAVS_CLEAN,
        "use_aug": False,
    }

    # exp02: Clean-50（Top-50 A/B 级）
    top50 = train_clean_ids[:50] if len(train_clean_ids) >= 50 else train_clean_ids
    experiments["exp02_clean50_r1"] = {
        "description": "Clean-50: 预处理后质量最优50条（A/B级）",
        "train_ids": top50,
        "test_ids": test_ids,
        "wav_dir": WAVS_CLEAN,
        "use_aug": False,
    }

    # exp03: Clean-80（过滤 C 级后的全量）
    top80 = train_clean_ids[:80] if len(train_clean_ids) >= 80 else train_clean_ids
    experiments["exp03_clean80_r1"] = {
        "description": f"Clean-80: 预处理后过滤C级，训练{len(top80)}条",
        "train_ids": top80,
        "test_ids": test_ids,
        "wav_dir": WAVS_CLEAN,
        "use_aug": False,
    }

    # exp04: Aug-50（clean50 + 语速/音高增强）
    experiments["exp04_aug50_r1"] = {
        "description": "Aug-50: clean50 + 语速±10% + 音高±50Hz增强",
        "train_ids": top50,
        "test_ids": test_ids,
        "wav_dir": WAVS_CLEAN,
        "use_aug": True,
    }

    # exp05: Full-Clean（全量清洗后，过滤 C 级）
    experiments["exp05_full_clean"] = {
        "description": f"Full-Clean: 全量{len(train_clean_ids)}条清洗+过滤C级后训练",
        "train_ids": train_clean_ids,
        "test_ids": test_ids,
        "wav_dir": WAVS_CLEAN,
        "use_aug": False,
    }

    # ── 生成 JSONL 文件 ──────────────────────────────────────
    aug_meta_path = BASE_DIR / "reports" / "augment_meta.json"

    print(f"\n{'─' * 60}")
    print(f"生成实验 JSONL 文件:")
    print(f"{'─' * 60}")

    summary = {}

    for exp_name in [
        "exp00_baseline_r1", "exp01_clean30_r1", "exp02_clean50_r1",
        "exp03_clean80_r1", "exp04_aug50_r1", "exp05_full_clean"
    ]:
        cfg = experiments[exp_name]
        exp_dir = EXPERIMENTS_DIR / exp_name
        output_train = exp_dir / "train_raw.jsonl"
        output_test = exp_dir / "test_raw.jsonl"

        train_ids = cfg["train_ids"]
        test_ids = cfg["test_ids"]
        wav_dir = cfg["wav_dir"]

        if cfg["use_aug"]:
            build_path, train_count = build_jsonl_with_aug(
                exp_name, train_ids, text_map, ref_audio, aug_meta_path, output_train
            )
        else:
            build_jsonl(exp_name, train_ids, text_map, wav_dir, ref_audio, output_train)
            train_count = len(train_ids)

        # 测试集用 clean wav（若无则 fallback 到原始）
        test_wav_dir = WAVS_CLEAN if (BASE_DIR / "wavs_clean").exists() else WAVS_ORIGINAL
        build_jsonl(
            f"{exp_name}_test", test_ids, text_map, test_wav_dir, ref_audio, output_test
        )

        summary[exp_name] = {
            "description": cfg["description"],
            "n_train": train_count,
            "n_test": len(test_ids),
            "wav_dir": wav_dir if not cfg["use_aug"] else f"{WAVS_CLEAN} + {WAVS_AUG}",
            "ref_audio": ref_audio,
            "train_jsonl": str(output_train),
            "test_jsonl": str(output_test),
        }

        print(f"  {exp_name}: train={train_count}, test={len(test_ids)} → {output_train}")

    # ── 保存汇总 ──────────────────────────────────────────────
    summary_path = EXPERIMENTS_DIR / "experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "test_ids": test_ids,
            "experiments": summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"实验分组完成！汇总: {summary_path}")
    print(f"{'=' * 60}")

    # ── 输出实验对照表 ────────────────────────────────────────
    print(f"\n  实验对照表:")
    print(f"  {'实验组':<22s} {'训练集':>6s} {'测试集':>6s} {'数据源':<30s}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*30}")
    for exp_name, s in summary.items():
        print(f"  {exp_name:<22s} {s['n_train']:>6d} {s['n_test']:>6d} {s['wav_dir']:<30s}")

    print(f"\n  下一步: 在每个实验目录运行 prepare_data.py 提取 audio_codes，然后 sft_12hz.py 训练")


if __name__ == "__main__":
    main()
