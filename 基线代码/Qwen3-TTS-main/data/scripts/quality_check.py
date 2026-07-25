#!/usr/bin/env python3
"""Phase 2: ASR 转写 + CER 打分 + A/B/C 分级。用法: python quality_check.py [v1|v2] [mps|cpu]"""

import re, json, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np; import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
ASR_MODEL_PATH = "/Users/wlf/.cache/modelscope/models/iic--speech_paraformer-vad-punc-zh"

VERSIONS = {
    "v1": {"input_dir": BASE_DIR / "wavs_clean", "fallback_dir": BASE_DIR / "wavs",
           "text_file": BASE_DIR / "metadata" / "text.txt",
           "report_out": BASE_DIR / "reports" / "quality_report_asr_v1.json", "label": "第一轮(自拟)"},
    "v2": {"input_dir": BASE_DIR / "wavs_v2_clean", "fallback_dir": BASE_DIR / "wavs_v2",
           "text_file": BASE_DIR / "metadata" / "text_v2.txt",
           "report_out": BASE_DIR / "reports" / "quality_report_asr_v2.json", "label": "第二轮(标贝)"},
}


@dataclass
class QualityResult:
    text_id: str; filename: str; ground_truth: str; asr_text: str
    cer: float; duration_sec: float; peak_db: float; rms_db: float
    silence_ratio: float; grade: str = ""; issues: list = None
    def __post_init__(self):
        if self.issues is None: self.issues = []


def load_text(path): m = {}; [m.__setitem__(g.group(1), g.group(2)) for g in [
    re.match(r"^(\d+)\s+(.+)$", l) for l in [l.strip() for l in open(path, encoding="utf-8")] if l] if g]; return m


def cer(ref, hyp):
    r, h = list(ref), list(hyp); n, m = len(r), len(h)
    if n == 0: return float(m)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[n][m]/n


def text_norm(t): import string; punc = string.punctuation + "，。！？、；：""''…—～《》（）【】「」"; return re.sub(r"\s+", "", t.translate(str.maketrans("", "", punc)))


def silence_ratio(y, sr, th_db=-40):
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-10: return 1.0
    th = rms * (10**(th_db/20)); fl, hl = int(sr*.025), int(sr*.01)
    nf = max(1, (len(y)-fl)//hl+1); return sum(1 for i in range(nf) if np.sqrt(np.mean(y[i*hl:i*hl+fl]**2))<th)/nf


def assign_grade(r):
    if r.cer<.10 and r.silence_ratio<.15: g="A"
    elif r.cer<.25 and r.silence_ratio<.30: g="B"
    else: g="C"
    if r.cer>.35: g="C"; r.issues.append(f"CER过高({r.cer:.0%})")
    if r.silence_ratio>.50: g="C"; r.issues.append(f"静音占比过高({r.silence_ratio:.0%})")
    if r.peak_db<-35: g=max(g,"B") if g=="A" else ("C" if g=="B" else g); r.issues.append("音量极低")
    r.grade = g; return r


def build_asr(device="mps"):
    from funasr import AutoModel
    return AutoModel(model=ASR_MODEL_PATH, device=device, disable_pbar=True)


def run(ver, device="mps"):
    cfg = VERSIONS[ver]
    audio_dir = cfg["input_dir"] if cfg["input_dir"].exists() and list(cfg["input_dir"].glob("*.wav")) else cfg["fallback_dir"]
    print(f"Phase 2: ASR质检 — {cfg['label']} | 音频: {audio_dir}")

    text_map = load_text(cfg["text_file"])
    asr = build_asr(device)
    stem_to_wav = {w.stem: w for w in sorted(audio_dir.glob("*.wav"))}
    results = []

    for stem in sorted(stem_to_wav, key=lambda s: int(s)):
        wav = stem_to_wav[stem]; text_id = str(int(stem)).zfill(6); gt = text_map.get(text_id, "")
        if not gt: continue
        try:
            y, sr = sf.read(str(wav)); dur = len(y)/sr
            peak = float(np.max(np.abs(y))); rms = np.sqrt(np.mean(y**2))
            peak_db = 20*np.log10(peak+1e-10); rms_db = 20*np.log10(rms+1e-10)
            sil = silence_ratio(y, sr)
            asr_out = asr.generate(input=str(wav)); asr_text = asr_out[0].get("text","") if isinstance(asr_out,list) and asr_out else ""
            c = cer(text_norm(gt), text_norm(asr_text))
            r = QualityResult(text_id, wav.name, gt, asr_text, round(c,4), round(dur,3), round(peak_db,1), round(rms_db,1), round(sil,3))
            r = assign_grade(r); results.append(r)
            print(f"  [{stem}] CER={r.cer:.1%} grade={r.grade}")
        except Exception as e:
            print(f"  [{stem}] ✗ {e}")
            results.append(QualityResult(text_id, wav.name, gt, "[ERROR]", 1.0, 0, -100, -100, 1.0, "C", [f"ASR异常:{str(e)[:80]}"]))

    if results:
        grades = {"A":0,"B":0,"C":0}; _ = [grades.update({r.grade: grades.get(r.grade,0)+1}) for r in results]
        cers = [r.cer for r in results]
        print(f"\n质检完成: A={grades['A']} B={grades['B']} C={grades['C']} | CER mean={np.mean(cers):.3f} median={np.median(cers):.3f}")

    report = {"version": ver, "asr_model": ASR_MODEL_PATH, "audio_source": str(audio_dir),
              "grade_distribution": {k: int(v) for k, v in grades.items()},
              "cer_stats": {"mean": round(float(np.mean(cers)),4), "median": round(float(np.median(cers)),4), "min": round(float(min(cers)),4), "max": round(float(max(cers)),4)} if results else {},
              "samples": sorted([{"text_id": r.text_id, "filename": r.filename, "ground_truth": r.ground_truth, "asr_text": r.asr_text, "cer": r.cer, "grade": r.grade, "duration_sec": r.duration_sec, "peak_db": r.peak_db, "rms_db": r.rms_db, "silence_ratio": r.silence_ratio, "issues": r.issues} for r in results], key=lambda x: x["cer"])}

    cfg["report_out"].parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["report_out"], "w", encoding="utf-8") as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {cfg['report_out']}")
    return 0


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if ver not in VERSIONS: print(f"未知版本: {ver}，可选: {list(VERSIONS)}"); sys.exit(1)
    dev = sys.argv[2] if len(sys.argv) > 2 else "mps"
    raise SystemExit(run(ver, dev))
