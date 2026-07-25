"""SIM Evaluation Pipeline"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SYSTEMS = ("zero_shot", "sft")




@dataclass(frozen=True)
class GeneratedSample:
    """一条待评测合成音频。"""

    system: str
    sample_id: str
    audio_path: Path


@dataclass(frozen=True)
class PairScore:
    """一条合成音频与一条真人 enrollment 音频的 SIM。"""

    system: str
    sample_id: str
    generated_path: Path
    enrollment_path: Path
    sim: float


Scorer = Callable[[Path, Path], float]


def _wav_files(directory: Path, label: str) -> list[Path]:
    """
    扫描目录
    排序后返回里面所有的.wav文件
    找不到就报错
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Can't find{label} directory:{directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: (path.stem.lower(), path.name.lower()),
    )
    if not files:
        raise ValueError(f"{label}No .wav files found in directory:{directory}")
    return files


def list_enrollment_audio(enrollment_dir: Path) -> list[Path]:
    """
    返回真人参考录音.wav
    """

    return _wav_files(enrollment_dir, "enrollment 真人参考音频")


def _index_by_sample_id(files: Iterable[Path], system: str) -> dict[str, Path]:
    """
    将.wav文件列表变成字典{样本名 → 文件路径},同时检测是否有重名
    """
    index: dict[str, Path] = {}
    for file_path in files:
        sample_id = file_path.stem
        if sample_id in index:
            raise ValueError(f"{system} 目录存在重复样本名：{sample_id}")
        index[sample_id] = file_path
    return index


def discover_generated_audio(generated_root: Path) -> list[GeneratedSample]:
    """发现 zero-shot 与 SFT 的音频，并验证两系统的样本名完全一致。"""

    indexes: dict[str, dict[str, Path]] = {}
    for system in SYSTEMS:
        files = _wav_files(generated_root / system, f"{system} 合成音频")
        indexes[system] = _index_by_sample_id(files, system)

    zero_shot_ids = set(indexes["zero_shot"])
    sft_ids = set(indexes["sft"])
    if zero_shot_ids != sft_ids:
        only_zero = ", ".join(sorted(zero_shot_ids - sft_ids)) or "无"
        only_sft = ", ".join(sorted(sft_ids - zero_shot_ids)) or "无"
        raise ValueError(
            "zero_shot 与 sft 样本名不一致："
            f"仅在 zero_shot 中：{only_zero}；仅在 sft 中：{only_sft}"
        )

    samples: list[GeneratedSample] = []
    for system in SYSTEMS:
        for sample_id in sorted(indexes[system]):
            samples.append(GeneratedSample(system, sample_id, indexes[system][sample_id]))
    return samples


def score_samples(
    samples: Iterable[GeneratedSample], enrollment_files: Iterable[Path], scorer: Scorer
) -> list[PairScore]:
    """
    每个合成语音与每个真人参考语音配对评分
    """

    enrollment = list(enrollment_files)
    if not enrollment:
        raise ValueError("至少需要一条 enrollment 真人参考音频")

    pair_scores: list[PairScore] = []
    for sample in samples:
        for enrollment_path in enrollment:
            score = float(scorer(sample.audio_path, enrollment_path))
            if not math.isfinite(score):
                raise ValueError(
                    f"评分结果不是有限数值：{sample.audio_path.name} 与 {enrollment_path.name}"
                )
            pair_scores.append(
                PairScore(
                    system=sample.system,
                    sample_id=sample.sample_id,
                    generated_path=sample.audio_path,
                    enrollment_path=enrollment_path,
                    sim=score,
                )
            )
    return pair_scores


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _population_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_scores(pair_scores: Iterable[PairScore]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """返回样本级和系统级 SIM 汇总行。"""

    grouped_pairs: dict[tuple[str, str, Path], list[PairScore]] = defaultdict(list)
    for item in pair_scores:
        grouped_pairs[(item.system, item.sample_id, item.generated_path)].append(item)
    if not grouped_pairs:
        raise ValueError("没有可汇总的配对评分")

    sample_rows: list[dict[str, object]] = []
    for (system, sample_id, generated_path), items in sorted(
        grouped_pairs.items(), key=lambda entry: (SYSTEMS.index(entry[0][0]), entry[0][1])
    ):
        scores = [item.sim for item in items]
        sample_rows.append(
            {
                "system": system,
                "sample_id": sample_id,
                "generated_audio": str(generated_path),
                "enrollment_count": len(items),
                "sim_mean": _mean(scores),
                "sim_std": _population_std(scores),
                "sim_min": min(scores),
                "sim_max": max(scores),
            }
        )

    values_by_system: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        values_by_system[str(row["system"])].append(float(row["sim_mean"]))

    summary_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        values = values_by_system.get(system, [])
        if not values:
            continue
        summary_rows.append(
            {
                "system": system,
                "sample_count": len(values),
                "sim_mean": _mean(values),
                "sim_std": _population_std(values),
                "sim_min": min(values),
                "sim_max": max(values),
            }
        )
    return sample_rows, summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results(pair_scores: Iterable[PairScore], output_dir: Path) -> dict[str, Path]:
    """将配对、样本和系统汇总分别写入 UTF-8 CSV。"""

    pair_score_list = list(pair_scores)
    sample_rows, summary_rows = summarize_scores(pair_score_list)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = [
        {
            "system": item.system,
            "sample_id": item.sample_id,
            "generated_audio": str(item.generated_path),
            "enrollment_audio": str(item.enrollment_path),
            "sim": item.sim,
        }
        for item in pair_score_list
    ]
    pair_path = output_dir / "pair_scores.csv"
    sample_path = output_dir / "sample_scores.csv"
    summary_path = output_dir / "summary.csv"
    _write_csv(
        pair_path,
        pair_rows,
        ["system", "sample_id", "generated_audio", "enrollment_audio", "sim"],
    )
    _write_csv(
        sample_path,
        sample_rows,
        [
            "system",
            "sample_id",
            "generated_audio",
            "enrollment_count",
            "sim_mean",
            "sim_std",
            "sim_min",
            "sim_max",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        ["system", "sample_count", "sim_mean", "sim_std", "sim_min", "sim_max"],
    )
    return {"pair_scores": pair_path, "sample_scores": sample_path, "summary": summary_path}
