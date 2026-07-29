#!/usr/bin/env python3
"""
批量 SIM 评测：对所有 14 组实验 × 5 epochs 的生成语音计算说话人相似度。

输出：
  data/reports/sim_batch_results.json  — 完整结果
  data/reports/sim_summary.csv         — 汇总表
"""

import json, csv, math, sys
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT / "data" / "experiments"
ENROLLMENT_DIR = PROJECT / "evaluation" / "data" / "enrollment"
OUT_DIR = PROJECT / "data" / "reports"

# 三类测试文本 → 文件名 stem 映射
TEST_TEXTS = {
    "seen_style":   "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long":  "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}

ERES_MODEL = "iic/speech_eres2netv2_sv_zh-cn_16k-common"


def flatten_embedding(value):
    if hasattr(value, "detach"): value = value.detach()
    if hasattr(value, "cpu"):    value = value.cpu()
    if hasattr(value, "tolist"): value = value.tolist()
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value: flat.extend(flatten_embedding(item))
        return flat
    return [float(value)]


def cosine(a, b):
    av, bv = flatten_embedding(a), flatten_embedding(b)
    dot = sum(x*y for x, y in zip(av, bv, strict=True))
    na = math.sqrt(sum(x*x for x in av))
    nb = math.sqrt(sum(x*x for x in bv))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def load_eres2net(device="cpu"):
    from funasr import AutoModel
    print(f"  加载 {ERES_MODEL} ...")
    return AutoModel(model=ERES_MODEL, device=device)


def extract_embedding(model, audio_path):
    result = model.generate(input=str(audio_path))
    return result[0]["spk_embedding"]


def run():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    print("=" * 60)
    print("批量 SIM 评测")
    print("=" * 60)

    # 加载 enrollment 音频
    enrollment_wavs = sorted(ENROLLMENT_DIR.glob("real_*.wav"))
    print(f"Enrollment: {len(enrollment_wavs)} 条真人参考")

    # 加载模型
    model = load_eres2net(device)
    print("  模型就绪\n")

    # 预提取 enrollment embeddings
    print("提取 enrollment embeddings...")
    enroll_embs = {}
    for wav in enrollment_wavs:
        enroll_embs[wav.stem] = extract_embedding(model, wav)
        print(f"  {wav.stem} ✓")
    print()

    # 扫描所有实验
    all_scores = []

    exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        if exp_name == "reports":
            continue

        inf_dir = exp_dir / "inference_samples"
        if not inf_dir.exists():
            print(f"  {exp_name}: 无 inference_samples，跳过")
            continue

        # Parse files: epoch0_seen_style.wav, epoch3_unseen_long.wav, etc.
        wavs = sorted(inf_dir.glob("*.wav"))
        if not wavs:
            continue

        print(f"  {exp_name}: {len(wavs)} wavs ", end="", flush=True)

        for wav in wavs:
            name = wav.stem  # "epoch0_seen_style"
            parts = name.split("_", 1)
            if len(parts) < 2:
                continue
            epoch_label = parts[0]        # "epoch0"
            text_label = parts[1]         # "seen_style"
            epoch_num = int(epoch_label.replace("epoch", ""))

            # Extract SFT embedding once
            sft_emb = extract_embedding(model, wav)

            # Compute vs each enrollment
            sims = []
            for enroll_stem, enroll_emb in enroll_embs.items():
                sim = cosine(sft_emb, enroll_emb)
                sims.append(sim)

            all_scores.append({
                "experiment": exp_name,
                "epoch": epoch_num,
                "text_type": text_label,
                "sft_wav": str(wav.name),
                "sim_mean": round(sum(sims)/len(sims), 6),
                "sim_max": round(max(sims), 6),
                "sim_min": round(min(sims), 6),
                "sim_individual": [round(s, 6) for s in sims],
            })
        print("✓")

    # ── 保存原始结果 ──
    raw_path = OUT_DIR / "sim_batch_results.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"enrollment_files": [w.name for w in enrollment_wavs],
                   "scores": all_scores}, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果: {raw_path} ({len(all_scores)} 条)")

    # ── 按实验汇总（取最佳 epoch 的均值）─-
    by_experiment = defaultdict(list)
    for s in all_scores:
        by_experiment[s["experiment"]].append(s)

    summary_rows = []
    for exp_name in sorted(by_experiment.keys()):
        scores = by_experiment[exp_name]
        sims = [s["sim_mean"] for s in scores]
        by_text = defaultdict(list)
        for s in scores:
            by_text[s["text_type"]].append(s["sim_mean"])

        # 每个 epoch 的平均
        by_epoch = defaultdict(list)
        for s in scores:
            by_epoch[s["epoch"]].append(s["sim_mean"])

        best_epoch = max(by_epoch, key=lambda e: sum(by_epoch[e]) / len(by_epoch[e]))

        summary_rows.append({
            "experiment": exp_name,
            "n_samples": len(scores),
            "sim_mean": round(sum(sims) / len(sims), 6),
            "sim_std": round((sum((x - sum(sims)/len(sims))**2 for x in sims) / len(sims))**0.5, 6) if len(sims) > 1 else 0,
            "sim_min": round(min(sims), 6),
            "sim_max": round(max(sims), 6),
            "best_epoch": best_epoch,
            "best_epoch_mean": round(sum(by_epoch[best_epoch]) / len(by_epoch[best_epoch]), 6),
            "seen_style_mean": round(sum(by_text.get("seen_style", [0])) / max(len(by_text.get("seen_style", [1])), 1), 6),
            "unseen_short_mean": round(sum(by_text.get("unseen_short", [0])) / max(len(by_text.get("unseen_short", [1])), 1), 6),
            "unseen_long_mean": round(sum(by_text.get("unseen_long", [0])) / max(len(by_text.get("unseen_long", [1])), 1), 6),
        })

    # ── 按 round 分组输出 ──
    for prefix, label in [("r1", "R1(自拟)"), ("r2", "R2(标贝)"), ("mixed", "混合")]:
        rows = [r for r in summary_rows if r["experiment"].startswith(prefix)]
        if not rows:
            continue
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {'实验组':<25s} {'样本':>5s} {'SIM均值':>8s} {'最优Epoch':>8s} {'seen':>7s} {'short':>7s} {'long':>7s}")
        print(f"  {'─'*25} {'─'*5} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")
        for r in rows:
            print(f"  {r['experiment']:<25s} {r['n_samples']:>5d} {r['sim_mean']:>8.4f} {r['best_epoch']:>8d} "
                  f"{r['seen_style_mean']:>7.4f} {r['unseen_short_mean']:>7.4f} {r['unseen_long_mean']:>7.4f}")

    # ── CSV 导出 ──
    csv_path = OUT_DIR / "sim_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["experiment", "n_samples", "sim_mean", "sim_std", "sim_min", "sim_max",
                  "best_epoch", "best_epoch_mean", "seen_style_mean", "unseen_short_mean", "unseen_long_mean"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nCSV 汇总: {csv_path}")

    # ── 关键对比总结 ──
    print(f"\n{'='*60}")
    print("  关键对比结论")
    print(f"  {'='*60}")
    # 找出每个 round 的最佳实验
    for prefix, label in [("r1", "R1(自拟)"), ("r2", "R2(标贝)")]:
        rows = [r for r in summary_rows if r["experiment"].startswith(prefix)]
        if rows:
            best = max(rows, key=lambda r: r["sim_mean"])
            print(f"  {label} 最佳: {best['experiment']} (SIM={best['sim_mean']:.4f}, epoch={best['best_epoch']})")

    if any(r["experiment"].startswith("mixed") for r in summary_rows):
        best_mix = max([r for r in summary_rows if r["experiment"].startswith("mixed")], key=lambda r: r["sim_mean"])
        print(f"  混合 最佳: {best_mix['experiment']} (SIM={best_mix['sim_mean']:.4f}, epoch={best_mix['best_epoch']})")

    print(f"\n  完成！")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
