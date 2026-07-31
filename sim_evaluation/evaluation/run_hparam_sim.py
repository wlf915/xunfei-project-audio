"""Run all hyperparameter SFT speaker-similarity comparisons in one process."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from evaluation.hparam_sim import (
    DEFAULT_MODEL_ID,
    build_eres2net_embedding_extractor,
    discover_candidates,
    discover_zero_shot,
    evaluate_candidates,
    validate_enrollment,
    write_hparam_results,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评测15组超参数/epoch SFT音频，并与固定zero-shot基线比较。"
    )
    parser.add_argument("--hparam-root", type=Path, required=True)
    parser.add_argument("--enrollment-dir", type=Path, required=True)
    parser.add_argument("--zero-shot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--local-model-path",
        type=Path,
        help="Optional local FunASR model directory; disables online update checks.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查目录、候选数量和样本ID，不加载模型、不写结果。",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: Callable[..., Any] | None = None,
) -> int:
    args = parse_args(argv)
    enrollment = validate_enrollment(args.enrollment_dir)
    zero_shot = discover_zero_shot(args.zero_shot_dir)
    candidates = discover_candidates(args.hparam_root)

    print(f"enrollment：{len(enrollment)}条（010/050/070）")
    print(f"zero-shot：{len(zero_shot)}条（test_010～test_200）")
    print(f"SFT候选：{len(candidates)}组，每组20条")

    if args.dry_run:
        print("dry-run验证通过：未加载模型，未写入结果。")
        return 0

    effective_factory = model_factory
    if args.local_model_path is not None and effective_factory is None:
        local_model_path = args.local_model_path.resolve()
        if not local_model_path.is_dir():
            raise ValueError(f"local model directory does not exist: {local_model_path}")

        def local_factory(**kwargs: Any) -> Any:
            from funasr import AutoModel

            return AutoModel(
                **kwargs,
                model_path=str(local_model_path),
                check_latest=False,
                disable_update=True,
            )

        effective_factory = local_factory

    extractor = build_eres2net_embedding_extractor(
        model_id=args.model_id,
        device=args.device,
        model_factory=effective_factory,
    )
    summary_rows, sample_rows, pair_rows = evaluate_candidates(
        zero_shot=zero_shot,
        candidates=candidates,
        enrollment=enrollment,
        embedding_extractor=extractor,
    )
    outputs = write_hparam_results(
        args.output_dir, summary_rows, sample_rows, pair_rows
    )
    metadata_path = args.output_dir / "run_metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hparam_root": str(args.hparam_root.resolve()),
        "enrollment_dir": str(args.enrollment_dir.resolve()),
        "enrollment_files": [path.name for path in enrollment],
        "zero_shot_dir": str(args.zero_shot_dir.resolve()),
        "zero_shot_reference_audio": "001.wav",
        "sft_reference_audio": "031.wav (inferred from run directory names)",
        "model_id": args.model_id,
        "local_model_path": (
            str(args.local_model_path.resolve())
            if args.local_model_path is not None
            else None
        ),
        "device": args.device,
        "candidate_count": len(candidates),
        "samples_per_condition": len(zero_shot),
        "enrollment_count": len(enrollment),
        "aggregation": (
            "For each generated sample, cosine similarity is averaged across "
            "the three enrollment embeddings; condition SIM is the mean across "
            "20 generated samples."
        ),
        "selection_caveat": (
            "Enrollment 010/050/070 was chosen after reviewing similarity "
            "behavior. Treat this output as sensitivity analysis rather than "
            "an unbiased primary comparison."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outputs["metadata"] = metadata_path

    best = max(summary_rows, key=lambda row: float(row["sft_mean"]))
    print(
        "最佳候选："
        f"lr={best['learning_rate']} epoch={best['epoch']} "
        f"SFT={float(best['sft_mean']):.6f} "
        f"zero-shot={float(best['zero_shot_mean']):.6f} "
        f"delta={float(best['delta']):+.6f}"
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
