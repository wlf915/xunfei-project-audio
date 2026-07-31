"""Batch speaker-similarity evaluation for hyperparameter SFT runs."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
import math
from pathlib import Path
import re
import statistics
from typing import Any


DEFAULT_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"

EXPECTED_ENROLLMENT_NAMES = ("010.wav", "050.wav", "070.wav")
EXPECTED_SAMPLE_IDS = tuple(f"test_{index:03d}" for index in range(10, 201, 10))
EXPECTED_EPOCHS = tuple(range(8))
EXPECTED_RUN_COUNT = 3

RUN_NAME_PATTERN = re.compile(
    r"^(?:[A-Za-z]+_)?(?:ep\d+_)?lr(?P<learning_rate>[^_]+)_"
    r"(?:(?:cosine\d+_)?ref031|per_epoch_audio)_seed\d+$"
)
EPOCH_NAME_PATTERN = re.compile(r"^epoch(?P<epoch>\d+)$")


@dataclass(frozen=True)
class HyperparamCandidate:
    run_name: str
    learning_rate: str
    epoch: int
    audio_by_id: dict[str, Path]


@dataclass(frozen=True)
class PairResult:
    system: str
    run_name: str
    learning_rate: str
    epoch: int | None
    sample_id: str
    generated_path: Path
    enrollment_path: Path
    sim: float


EmbeddingExtractor = Callable[[Path], Any]


def _flatten_embedding(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_embedding(item))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError) as error:
        raise RuntimeError("Speaker embedding is not numeric") from error


def _cosine_similarity(first: Any, second: Any) -> float:
    first_values = _flatten_embedding(first)
    second_values = _flatten_embedding(second)
    if not first_values or len(first_values) != len(second_values):
        raise RuntimeError("Speaker embeddings are empty or have different dimensions")
    dot_product = sum(
        left * right for left, right in zip(first_values, second_values, strict=True)
    )
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    if first_norm == 0.0 or second_norm == 0.0:
        raise RuntimeError("Speaker embedding norm is zero")
    return dot_product / (first_norm * second_norm)


def cache_embedding_extractor(extractor: EmbeddingExtractor) -> EmbeddingExtractor:
    @cache
    def extract_resolved(resolved_path: Path) -> Any:
        return extractor(resolved_path)

    def extract(audio_path: Path) -> Any:
        return extract_resolved(Path(audio_path).resolve())

    return extract


def score_audio_map(
    *,
    system: str,
    run_name: str,
    learning_rate: str,
    epoch: int | None,
    audio_by_id: dict[str, Path],
    enrollment: Iterable[Path],
    embedding_extractor: EmbeddingExtractor,
) -> tuple[list[PairResult], dict[str, float]]:
    enrollment_paths = list(enrollment)
    if not enrollment_paths:
        raise ValueError("At least one enrollment audio is required")

    pair_rows: list[PairResult] = []
    sample_scores: dict[str, float] = {}
    for sample_id in EXPECTED_SAMPLE_IDS:
        generated_path = audio_by_id[sample_id]
        generated_embedding = embedding_extractor(generated_path)
        scores: list[float] = []
        for enrollment_path in enrollment_paths:
            score = float(
                _cosine_similarity(
                    generated_embedding, embedding_extractor(enrollment_path)
                )
            )
            if not math.isfinite(score):
                raise RuntimeError(
                    f"SIM is not finite: {generated_path} vs {enrollment_path}"
                )
            scores.append(score)
            pair_rows.append(
                PairResult(
                    system=system,
                    run_name=run_name,
                    learning_rate=learning_rate,
                    epoch=epoch,
                    sample_id=sample_id,
                    generated_path=generated_path,
                    enrollment_path=enrollment_path,
                    sim=score,
                )
            )
        sample_scores[sample_id] = statistics.fmean(scores)
    return pair_rows, sample_scores


def evaluate_candidates(
    *,
    zero_shot: dict[str, Path],
    candidates: Iterable[HyperparamCandidate],
    enrollment: Iterable[Path],
    embedding_extractor: EmbeddingExtractor,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[PairResult]]:
    enrollment_paths = list(enrollment)
    cached_extractor = cache_embedding_extractor(embedding_extractor)
    zero_pairs, zero_scores = score_audio_map(
        system="zero_shot",
        run_name="baseline",
        learning_rate="",
        epoch=None,
        audio_by_id=zero_shot,
        enrollment=enrollment_paths,
        embedding_extractor=cached_extractor,
    )
    zero_mean = statistics.fmean(zero_scores.values())

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    pair_rows = list(zero_pairs)
    for candidate in candidates:
        candidate_pairs, sft_scores = score_audio_map(
            system="sft",
            run_name=candidate.run_name,
            learning_rate=candidate.learning_rate,
            epoch=candidate.epoch,
            audio_by_id=candidate.audio_by_id,
            enrollment=enrollment_paths,
            embedding_extractor=cached_extractor,
        )
        pair_rows.extend(candidate_pairs)
        sft_values = [sft_scores[sample_id] for sample_id in EXPECTED_SAMPLE_IDS]
        deltas = [
            sft_scores[sample_id] - zero_scores[sample_id]
            for sample_id in EXPECTED_SAMPLE_IDS
        ]
        wins = sum(delta > 1e-12 for delta in deltas)
        losses = sum(delta < -1e-12 for delta in deltas)
        ties = len(deltas) - wins - losses
        sft_mean = statistics.fmean(sft_values)
        summary_rows.append(
            {
                "run_name": candidate.run_name,
                "learning_rate": candidate.learning_rate,
                "epoch": candidate.epoch,
                "sample_count": len(EXPECTED_SAMPLE_IDS),
                "zero_shot_mean": zero_mean,
                "sft_mean": sft_mean,
                "delta": sft_mean - zero_mean,
                "sft_wins": wins,
                "sft_losses": losses,
                "ties": ties,
                "sft_std": statistics.pstdev(sft_values),
                "sft_min": min(sft_values),
                "sft_max": max(sft_values),
            }
        )
        for sample_id in EXPECTED_SAMPLE_IDS:
            sample_rows.append(
                {
                    "run_name": candidate.run_name,
                    "learning_rate": candidate.learning_rate,
                    "epoch": candidate.epoch,
                    "sample_id": sample_id,
                    "zero_shot_mean": zero_scores[sample_id],
                    "sft_mean": sft_scores[sample_id],
                    "delta": sft_scores[sample_id] - zero_scores[sample_id],
                }
            )

    summary_rows.sort(
        key=lambda row: (float(str(row["learning_rate"])), int(row["epoch"]))
    )
    return summary_rows, sample_rows, pair_rows


def _require_directory(directory: Path, label: str) -> Path:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{label}目录不存在：{directory}")
    return directory


def _top_level_wav_index(directory: Path, label: str) -> dict[str, Path]:
    directory = _require_directory(directory, label)
    index: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        sample_id = path.stem
        if sample_id in index:
            raise ValueError(f"{label}存在重复样本ID：{sample_id}")
        index[sample_id] = path
    return index


def _select_expected_samples(directory: Path, label: str) -> dict[str, Path]:
    index = _top_level_wav_index(directory, label)
    missing = [sample_id for sample_id in EXPECTED_SAMPLE_IDS if sample_id not in index]
    if missing:
        raise ValueError(f"{label}缺少必需样本：{', '.join(missing)}")
    return {sample_id: index[sample_id] for sample_id in EXPECTED_SAMPLE_IDS}


def validate_enrollment(directory: Path) -> list[Path]:
    index = _top_level_wav_index(directory, "enrollment")
    actual_names = {path.name.lower() for path in index.values()}
    expected_names = set(EXPECTED_ENROLLMENT_NAMES)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        expected_display = "、".join(EXPECTED_ENROLLMENT_NAMES)
        raise ValueError(
            f"enrollment必须且只能包含{expected_display}；"
            f"缺少：{', '.join(missing) or '无'}；"
            f"多余：{', '.join(extra) or '无'}"
        )
    by_name = {path.name.lower(): path for path in index.values()}
    return [by_name[name] for name in EXPECTED_ENROLLMENT_NAMES]


def discover_zero_shot(directory: Path) -> dict[str, Path]:
    return _select_expected_samples(Path(directory), "zero-shot")


def discover_candidates(hparam_root: Path) -> list[HyperparamCandidate]:
    hparam_root = _require_directory(Path(hparam_root), "hparam_runs")
    run_entries: list[tuple[float, str, Path]] = []
    for path in hparam_root.iterdir():
        if not path.is_dir():
            continue
        match = RUN_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        learning_rate = match.group("learning_rate")
        try:
            numeric_learning_rate = float(learning_rate)
        except ValueError as error:
            raise ValueError(f"无法解析学习率目录名：{path.name}") from error
        run_entries.append((numeric_learning_rate, learning_rate, path))

    run_entries.sort(key=lambda item: item[0])
    if len(run_entries) != EXPECTED_RUN_COUNT:
        raise ValueError(
            f"应发现{EXPECTED_RUN_COUNT}个超参数run，实际发现{len(run_entries)}个："
            f"{', '.join(path.name for _, _, path in run_entries) or '无'}"
        )

    candidates: list[HyperparamCandidate] = []
    for _, learning_rate, run_path in run_entries:
        test_root = _require_directory(
            run_path / "results" / "test20", f"{run_path.name} test20"
        )
        epoch_paths: dict[int, Path] = {}
        for path in test_root.iterdir():
            if not path.is_dir():
                continue
            match = EPOCH_NAME_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            epoch = int(match.group("epoch"))
            if epoch in epoch_paths:
                raise ValueError(f"{run_path.name}存在重复epoch：epoch{epoch}")
            epoch_paths[epoch] = path

        actual_epochs = set(epoch_paths)
        expected_epochs = set(EXPECTED_EPOCHS)
        if actual_epochs != expected_epochs:
            missing = sorted(expected_epochs - actual_epochs)
            extra = sorted(actual_epochs - expected_epochs)
            raise ValueError(
                f"{run_path.name} epoch目录不完整；"
                f"缺少：{missing or '无'}；多余：{extra or '无'}"
            )

        for epoch in EXPECTED_EPOCHS:
            audio_by_id = _select_expected_samples(
                epoch_paths[epoch], f"{run_path.name}/epoch{epoch}"
            )
            candidates.append(
                HyperparamCandidate(
                    run_name=run_path.name,
                    learning_rate=learning_rate,
                    epoch=epoch,
                    audio_by_id=audio_by_id,
                )
            )
    return candidates


def build_eres2net_embedding_extractor(
    *,
    model_id: str,
    device: str,
    model_factory: Callable[..., Any] | None = None,
) -> EmbeddingExtractor:
    if model_factory is None:
        try:
            from funasr import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "FunASR is required. Install evaluation/requirements.txt first."
            ) from error
        model_factory = AutoModel

    verifier = model_factory(model=model_id, device=device)

    def extract(audio_path: Path) -> Any:
        result = verifier.generate(input=str(audio_path))
        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise RuntimeError(f"ERes2Net returned an invalid result for {audio_path}")
        if "spk_embedding" not in result[0]:
            raise RuntimeError(f"ERes2Net returned no spk_embedding for {audio_path}")
        return result[0]["spk_embedding"]

    return extract


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_hparam_results(
    output_dir: Path,
    summary_rows: Iterable[dict[str, object]],
    sample_rows: Iterable[dict[str, object]],
    pair_rows: Iterable[PairResult],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    summaries = list(summary_rows)
    samples = list(sample_rows)
    pairs = list(pair_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "hparam_summary.csv"
    sample_path = output_dir / "hparam_sample_scores.csv"
    pair_path = output_dir / "hparam_pair_scores.csv"

    _write_csv(
        summary_path,
        summaries,
        [
            "run_name",
            "learning_rate",
            "epoch",
            "sample_count",
            "zero_shot_mean",
            "sft_mean",
            "delta",
            "ties",
            "sft_std",
            "sft_min",
            "sft_max",
        ],
    )
    _write_csv(
        sample_path,
        samples,
        [
            "run_name",
            "learning_rate",
            "epoch",
            "sample_id",
            "zero_shot_mean",
            "sft_mean",
            "delta",
        ],
    )
    pair_dicts = [
        {
            "system": row.system,
            "run_name": row.run_name,
            "learning_rate": row.learning_rate,
            "epoch": "" if row.epoch is None else row.epoch,
            "sample_id": row.sample_id,
            "generated_audio": str(row.generated_path),
            "enrollment_audio": str(row.enrollment_path),
            "sim": row.sim,
        }
        for row in pairs
    ]
    _write_csv(
        pair_path,
        pair_dicts,
        [
            "system",
            "run_name",
            "learning_rate",
            "epoch",
            "sample_id",
            "generated_audio",
            "enrollment_audio",
            "sim",
        ],
    )
    return {"summary": summary_path, "samples": sample_path, "pairs": pair_path}

