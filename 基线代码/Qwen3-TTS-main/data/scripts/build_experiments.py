#!/usr/bin/env python3
"""
Phase 4: 构建双轮实验分组 — v1/v2/混合 共 14 组消融实验

用法: python build_experiments.py [v1|v2|all]
  v1   — 仅重建 R1 6组
  v2   — 仅重建 R2 6组
  all  — 双轮 + 混合 (默认)
"""

import re, json, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"

V1 = {
    "text_file": BASE_DIR / "metadata" / "text.txt",
    "quality_report": BASE_DIR / "reports" / "quality_report_asr_v1.json",
    "aug_meta": BASE_DIR / "reports" / "augment_meta_v1.json",
    "wav_raw": "data/wavs", "wav_clean": "data/wavs_clean", "wav_aug": "data/wavs_augmented",
    "round_label": "R1(自拟)", "prefix": "r1",
}
V2 = {
    "text_file": BASE_DIR / "metadata" / "text_v2.txt",
    "quality_report": BASE_DIR / "reports" / "quality_report_asr_v2.json",
    "aug_meta": BASE_DIR / "reports" / "augment_meta_v2.json",
    "wav_raw": "data/wavs_v2", "wav_clean": "data/wavs_v2_clean", "wav_aug": "data/wavs_v2_augmented",
    "round_label": "R2(标贝)", "prefix": "r2",
}


def load_text(path): m = {}; [m.__setitem__(g.group(1), g.group(2)) for g in [
    re.match(r"^(\d+)\s+(.+)$", l) for l in [l.strip() for l in open(path, encoding="utf-8")] if l] if g]; return m


def is_test(tid): return int(tid) % 10 == 0


def load_quality(path):
    if not path.exists(): return {}
    with open(path, encoding="utf-8") as f: report = json.load(f)
    return {s["text_id"]: s for s in report.get("samples", [])}


def load_aug_map(path):
    if not path.exists(): return {}
    with open(path, encoding="utf-8") as f: meta = json.load(f)
    m = {}
    for e in meta.get("entries", []):
        m[e["orig_stem"]] = e.get("augmented", [])
    return m


def select_ref(quality_index, text_map, cfg):
    """选最佳参考音频: A级、CER最低"""
    a_samples = []
    for tid, q in quality_index.items():
        if tid not in text_map: continue
        if q.get("grade") == "A" and 3.0 <= q.get("duration_sec", 0) <= 6.0:
            a_samples.append((tid, q))
    if not a_samples:
        a_samples = [(tid, q) for tid, q in quality_index.items()
                     if tid in text_map and q.get("grade") in ("A", "B") and q.get("duration_sec", 1) > 0]
    if not a_samples:
        # fallback
        ref_stem = "001"; ref_path = f"{cfg['wav_clean']}/{ref_stem}.wav"
        return ref_path, text_map.get(f"{int(ref_stem):06d}", "")
    sort_key = "cer" if "cer" in a_samples[0][1] else "score"
    a_samples.sort(key=lambda x: x[1][sort_key], reverse=(sort_key == "score"))
    best_tid = a_samples[0][0]; best_stem = str(int(best_tid)).zfill(3)
    return f"{cfg['wav_clean']}/{best_stem}.wav", text_map.get(best_tid, "")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_round(cfg, exp_offset, round_name):
    """为一个数据轮生成 6 组实验 JSONL，返回 summary"""
    text_map = load_text(cfg["text_file"])
    quality_index = load_quality(cfg["quality_report"])
    aug_map = load_aug_map(cfg["aug_meta"])

    all_ids = sorted(text_map.keys())
    train_ids = [tid for tid in all_ids if not is_test(tid)]
    test_ids = [tid for tid in all_ids if is_test(tid)]

    ref_audio_rel, ref_text = select_ref(quality_index, text_map, cfg)

    # 按质量排序，过滤 C 级
    if quality_index:
        sort_key = "cer" if "cer" in next(iter(quality_index.values())) else "score"
        reverse = (sort_key == "score")
        ranked = sorted(
            [(tid, quality_index[tid]) for tid in train_ids if tid in quality_index],
            key=lambda x: x[1][sort_key], reverse=reverse)
        clean_ids = [tid for tid, _ in ranked if quality_index[tid].get("grade") in ("A", "B")]
        sorted_ids = [tid for tid, _ in ranked]
    else:
        clean_ids = train_ids; sorted_ids = train_ids

    top30 = clean_ids[:min(30, len(clean_ids))]
    top50 = clean_ids[:min(50, len(clean_ids))]
    top80 = clean_ids[:min(80, len(clean_ids))]

    experiments = {
        f"{cfg['prefix']}_{exp_offset:02d}_baseline": (train_ids, cfg["wav_raw"], False, f"{round_name} 基线(原始)"),
        f"{cfg['prefix']}_{exp_offset+1:02d}_clean30": (top30, cfg["wav_clean"], False, f"{round_name} Clean-30"),
        f"{cfg['prefix']}_{exp_offset+2:02d}_clean50": (top50, cfg["wav_clean"], False, f"{round_name} Clean-50"),
        f"{cfg['prefix']}_{exp_offset+3:02d}_clean80": (top80, cfg["wav_clean"], False, f"{round_name} Clean-80"),
        f"{cfg['prefix']}_{exp_offset+4:02d}_aug50":   (top50, cfg["wav_clean"], True,  f"{round_name} Aug-50"),
        f"{cfg['prefix']}_{exp_offset+5:02d}_full_clean": (clean_ids, cfg["wav_clean"], False, f"{round_name} Full-Clean"),
    }

    summary = {}
    for exp_name, (tids, wav_dir, use_aug, desc) in experiments.items():
        exp_dir = EXPERIMENTS_DIR / exp_name
        records = []
        for tid in sorted(tids, key=lambda x: int(x)):
            stem = str(int(tid)).zfill(3); text = text_map.get(tid, "")
            records.append({"audio": f"{wav_dir}/{stem}.wav", "text": text, "ref_audio": ref_audio_rel})
            if use_aug and stem in aug_map:
                for v in aug_map[stem]:
                    records.append({"audio": f"{cfg['wav_aug']}/{v['filename']}", "text": text, "ref_audio": ref_audio_rel})

        write_jsonl(exp_dir / "train_raw.jsonl", records)
        # test set — 用 clean wav，不存在时回退 raw
        test_wav_dir = cfg["wav_clean"] if (BASE_DIR / cfg["wav_clean"].split("/", 1)[1]).exists() else cfg["wav_raw"]
        test_records = [{"audio": f"{test_wav_dir}/{str(int(tid)).zfill(3)}.wav", "text": text_map.get(tid, ""), "ref_audio": ref_audio_rel} for tid in sorted(test_ids, key=lambda x: int(x))]
        write_jsonl(exp_dir / "test_raw.jsonl", test_records)

        summary[exp_name] = {"desc": desc, "train": len(records), "test": len(test_ids)}
        print(f"  {exp_name}: train={len(records)} test={len(test_ids)}  ref={ref_audio_rel.split('/')[-1]}")

    return summary, ref_audio_rel, ref_text, text_map, quality_index, clean_ids, test_ids, top50


def build_mixed(v1_cfg, v2_cfg, v1_clean, v2_clean, test_ids, v1_map, v2_map, v1_ref, v2_ref):
    """构建 v1+v2 混合实验"""
    summary = {}

    # Mixed-1: v1 Top-25 + v2 Top-25（不同轮次数据，即使 ID 重叠也保留）
    # 用 (round, tid) 区分
    test_ids_r1 = [tid for tid in sorted(v1_map.keys()) if is_test(tid)]
    test_recs = [{"audio": f"{v1_cfg['wav_clean']}/{str(int(tid)).zfill(3)}.wav", "text": v1_map.get(tid, ""), "ref_audio": v1_ref} for tid in test_ids_r1]

    for mixed_name, v1_ids, v2_ids in [
        ("mixed_01_top50", v1_clean[:25], v2_clean[:25]),
        ("mixed_02_all", v1_clean, v2_clean),
    ]:
        records = []
        seen = set()
        # v1 entries
        for tid in v1_ids:
            stem = str(int(tid)).zfill(3); text = v1_map[tid]
            wav = f"{v1_cfg['wav_clean']}/{stem}.wav"
            records.append({"audio": wav, "text": text, "ref_audio": v1_ref})
            seen.add(("v1", tid))
        # v2 entries（独立加入，即使 ID 重复也是不同文本）
        for tid in v2_ids:
            stem = str(int(tid)).zfill(3); text = v2_map.get(tid, "")
            wav = f"{v2_cfg['wav_clean']}/{stem}.wav"
            records.append({"audio": wav, "text": text, "ref_audio": v2_ref})
        exp_dir = EXPERIMENTS_DIR / mixed_name
        write_jsonl(exp_dir / "train_raw.jsonl", records)
        write_jsonl(exp_dir / "test_raw.jsonl", test_recs)
        desc = ("混合 Top-50 (R1-25+R2-25)" if "top50" in mixed_name
                else f"混合全量 A/B (R1×{len(v1_ids)}+R2×{len(v2_ids)})={len(records)}条")
        summary[mixed_name] = {"desc": desc, "train": len(records), "test": len(test_ids_r1)}
        print(f"  {mixed_name}: train={len(records)} test={len(test_ids_r1)}")

    return summary


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    v1_clean, v2_clean, test_ids, v1_map, v2_map, v1_ref, v2_ref = None, None, None, None, None, None, None

    if mode in ("v1", "all"):
        print(f"=== R1 实验分组 ===")
        s1, v1_ref, _, v1_map, v1_qi, v1_clean, test_ids, v1_top50 = build_round(V1, 0, "R1(自拟)")
        all_summary.update(s1)
        # gen top50 IDs for v1
        top50_file = BASE_DIR / "metadata" / "top50_ids_v1.txt"
        with open(top50_file, "w") as f:
            for tid in v1_top50: f.write(tid + "\n")

    if mode in ("v2", "all"):
        print(f"\n=== R2 实验分组 ===")
        s2, v2_ref, _, v2_map, v2_qi, v2_clean, _, v2_top50 = build_round(V2, 10, "R2(标贝)")
        all_summary.update(s2)
        top50_file = BASE_DIR / "metadata" / "top50_ids_v2.txt"
        with open(top50_file, "w") as f:
            for tid in v2_top50: f.write(tid + "\n")
        if mode == "v2": test_ids = [tid for tid in sorted(v2_map.keys()) if is_test(tid)]

    if mode == "all" and v1_clean is not None and v2_clean is not None:
        print(f"\n=== 混合实验分组 ===")
        # R1 test_ids 和 R2 test_ids 可能不同，统一用 R1 的
        test_ids_r1 = [tid for tid in sorted(v1_map.keys()) if is_test(tid)]
        s3 = build_mixed(V1, V2, v1_clean, v2_clean, test_ids_r1, v1_map, v2_map, v1_ref, v2_ref)
        all_summary.update(s3)

    # 保存汇总
    summary_path = EXPERIMENTS_DIR / "experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"实验分组完成! 共 {len(all_summary)} 组")
    print(f"{'实验组':<30s} {'训练':>6s} {'测试':>6s} {'说明'}")
    print(f"{'─'*30} {'─'*6} {'─'*6} {'─'*40}")
    for name, s in all_summary.items():
        print(f"  {name:<28s} {s['train']:>6d} {s['test']:>6d} {s['desc']}")
    print(f"\n汇总: {summary_path}")


if __name__ == "__main__":
    main()
