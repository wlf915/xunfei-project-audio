#!/usr/bin/env python3
"""ASR CER 评测：对所有实验生成语音做 Paraformer 转写 → 计算 CER"""
import json, csv, sys, re
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT / "data" / "experiments"
OUT_DIR = PROJECT / "data" / "reports"

TEST_TEXTS = {
    "seen_style":   "如果时间允许，我们下午一起去实验室继续完成模型训练。",
    "unseen_short": "这是使用我的个人语音数据微调后的测试语音。",
    "unseen_long":  "人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。",
}

ASR_MODEL = "/Users/wlf/.cache/modelscope/models/iic--speech_paraformer-vad-punc-zh"


def cer(ref, hyp):
    r, h = list(ref), list(hyp); n, m = len(r), len(h)
    if n == 0: return float(m)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[n][m] / n


def text_norm(t):
    import string
    punc = string.punctuation + "，。！？、；：""''…—～《》（）【】「」"
    return re.sub(r"\s+", "", t.translate(str.maketrans("", "", punc)))


def run():
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    print("=" * 60)
    print("批量 CER 评测（ASR Paraformer）")
    print("=" * 60)

    from funasr import AutoModel
    print(f"  加载 ASR 模型: {ASR_MODEL}")
    asr = AutoModel(model=ASR_MODEL, device=device, disable_pbar=True)
    print("  模型就绪\n")

    all_scores = []
    exp_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        if exp_name == "reports":
            continue

        inf_dir = exp_dir / "inference_samples"
        if not inf_dir.exists():
            continue

        wavs = sorted(inf_dir.glob("*.wav"))
        if not wavs:
            continue

        print(f"  {exp_name}: {len(wavs)} wavs ", end="", flush=True)

        for wav in wavs:
            name = wav.stem
            parts = name.split("_", 1)
            if len(parts) < 2: continue
            epoch_label = parts[0]
            text_label = parts[1]
            epoch_num = int(epoch_label.replace("epoch", ""))
            gt = TEST_TEXTS.get(text_label, "")
            if not gt: continue

            result = asr.generate(input=str(wav))
            asr_text = result[0].get("text", "") if isinstance(result, list) and result else ""

            c = cer(text_norm(gt), text_norm(asr_text))
            all_scores.append({
                "experiment": exp_name, "epoch": epoch_num,
                "text_type": text_label, "wav": str(wav.name),
                "ground_truth": gt, "asr_text": asr_text, "cer": round(c, 6),
            })
        print("✓")

    # 保存原始结果
    raw_path = OUT_DIR / "cer_batch_results.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"scores": all_scores}, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果: {raw_path} ({len(all_scores)} 条)")

    # 按实验汇总
    by_experiment = defaultdict(list)
    for s in all_scores: by_experiment[s["experiment"]].append(s)

    summary_rows = []
    for exp_name in sorted(by_experiment.keys()):
        scores = by_experiment[exp_name]
        cers = [s["cer"] for s in scores]
        by_epoch = defaultdict(list)
        for s in scores: by_epoch[s["epoch"]].append(s["cer"])
        best_epoch = min(by_epoch, key=lambda e: sum(by_epoch[e])/len(by_epoch[e]))
        by_text = defaultdict(list)
        for s in scores: by_text[s["text_type"]].append(s["cer"])

        summary_rows.append({
            "experiment": exp_name, "n_samples": len(scores),
            "cer_mean": round(sum(cers)/len(cers), 6),
            "cer_min": round(min(cers), 6), "cer_max": round(max(cers), 6),
            "best_epoch": best_epoch,
            "best_epoch_cer": round(sum(by_epoch[best_epoch])/len(by_epoch[best_epoch]), 6),
            "seen_cer": round(sum(by_text.get("seen_style",[0]))/max(len(by_text.get("seen_style",[1])),1), 6),
            "short_cer": round(sum(by_text.get("unseen_short",[0]))/max(len(by_text.get("unseen_short",[1])),1), 6),
            "long_cer": round(sum(by_text.get("unseen_long",[0]))/max(len(by_text.get("unseen_long",[1])),1), 6),
        })

    for prefix, label in [("r1", "R1(自拟)"), ("r2", "R2(标贝)"), ("mixed", "混合")]:
        rows = [r for r in summary_rows if r["experiment"].startswith(prefix)]
        if not rows: continue
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {'实验组':<25s} {'样本':>5s} {'CER均值':>9s} {'最优Epoch':>8s} {'seen':>8s} {'short':>8s} {'long':>8s}")
        print(f"  {'─'*25} {'─'*5} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for r in rows:
            print(f"  {r['experiment']:<25s} {r['n_samples']:>5d} {r['cer_mean']:>8.4f} {r['best_epoch']:>8d} "
                  f"{r['seen_cer']:>8.4f} {r['short_cer']:>8.4f} {r['long_cer']:>8.4f}")

    csv_path = OUT_DIR / "cer_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["experiment","n_samples","cer_mean","cer_min","cer_max",
                                           "best_epoch","best_epoch_cer","seen_cer","short_cer","long_cer"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"\nCSV: {csv_path}")
    print("完成!")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
