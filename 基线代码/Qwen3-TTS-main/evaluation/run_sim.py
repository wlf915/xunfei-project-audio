"""批量计算zero_shot合成语音与SFT合成语音的SIM"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any, Callable


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from evaluation.sim_pipeline import (
    PairScore,
    Scorer,
    discover_generated_audio,
    list_enrollment_audio,
    score_samples,
    summarize_scores,
    write_results,
)


DEFAULT_MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用中文 ERes2NetV2 模型自动评测 zero-shot 与 SFT 合成语音的 SIM。"
    )
    parser.add_argument(
        "--enrollment-dir",
        type=Path,
        required=True,
        help="独立真人参考录音目录，至少包含一条 WAV。",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="包含 zero_shot/ 和 sft/ 两个子目录的合成音频目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="CSV 和对比图输出目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="FunASR 推理设备，例如 cuda:0 或 cpu（默认：cuda:0）。",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"ERes2Net 说话人验证模型（默认：{DEFAULT_MODEL_ID}）。",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="只输出CSV, 不生成可视化图表",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证目录与样本名，不下载模型、不评分、不写结果。",
    )
    return parser.parse_args(argv)


def _flatten_embedding(value: Any) -> list[float]:
    """将 FunASR 返回的 list、NumPy 或 Torch embedding 展平成浮点列表。"""

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
        raise RuntimeError("ERes2Net 返回的 spk_embedding 不是数值向量") from error


def _cosine_similarity(first: Any, second: Any) -> float:
    first_values = _flatten_embedding(first)
    second_values = _flatten_embedding(second)
    if not first_values or len(first_values) != len(second_values):
        raise RuntimeError("两条 spk_embedding 为空或维度不一致，无法计算 SIM")
    dot_product = sum(left * right for left, right in zip(first_values, second_values, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    if first_norm == 0.0 or second_norm == 0.0:
        raise RuntimeError("spk_embedding范数为0,无法计算 SIM")
    return dot_product / (first_norm * second_norm)


def build_eres2net_scorer(
    model_id: str,
    device: str,
    model_factory: Callable[..., Any] | None = None,
) -> Scorer:
    """Load ERes2NetV2, 将两段语音的 embedding 转换为余弦 SIM"""

    if model_factory is None:
        try:
            from funasr import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "缺funasr Not Found"
                "python -m pip install -r evaluation/requirements.txt"
            ) from error
        model_factory = AutoModel

    verifier = model_factory(model=model_id, device=device)

    def extract_embedding(audio_path: Path) -> Any:
        result = verifier.generate(input=str(audio_path))
        if not isinstance(result, list) or not result or "spk_embedding" not in result[0]:
            raise RuntimeError(
                "ERes2Net未返回spk_embedding。请检查音频是否可读, 以及 FunASR 与模型版本是否匹配。"
            )
        return result[0]["spk_embedding"]

    def scorer(generated_audio: Path, enrollment_audio: Path) -> float:
        return _cosine_similarity(extract_embedding(generated_audio), extract_embedding(enrollment_audio))

    return scorer


def create_comparison_plot(sample_rows: Sequence[dict[str, object]], output_path: Path) -> None:
    """绘制每个系统的样本级 SIM 分布及平均值。"""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values_by_system: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        values_by_system[str(row["system"])].append(float(row["sim_mean"]))
    systems = [system for system in ("zero_shot", "sft") if values_by_system[system]]
    if not systems:
        raise ValueError("没有可绘图的样本级SIM结果")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.8))
    values = [values_by_system[system] for system in systems]
    box_plot = axis.boxplot(values, tick_labels=systems, patch_artist=True, showmeans=True)
    colors = ["#4C78A8", "#F58518"]
    for patch, color in zip(box_plot["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for index, system in enumerate(systems, start=1):
        system_values = values_by_system[system]
        offsets = [((item % 7) - 3) * 0.025 for item in range(len(system_values))]
        axis.scatter(
            [index + offset for offset in offsets],
            system_values,
            color=colors[index - 1],
            s=30,
            alpha=0.8,
            label="sample SIM" if index == 1 else None,
        )
    axis.set_title("Speaker Similarity (SIM) Comparison")
    axis.set_ylabel("SIM score")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _print_summary(pair_scores: Sequence[PairScore]) -> list[dict[str, object]]:
    _, summary_rows = summarize_scores(pair_scores)
    print("\nSIM 汇总结果：")
    for row in summary_rows:
        print(
            f"  {row['system']}: n={row['sample_count']}, "
            f"mean={float(row['sim_mean']):.4f}, std={float(row['sim_std']):.4f}"
        )
    return summary_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enrollment_files = list_enrollment_audio(args.enrollment_dir)
    samples = discover_generated_audio(args.generated_dir)
    print(f"已发现 {len(enrollment_files)} 条 enrollment 真人参考录音。")
    print(f"已发现 {len(samples) // 2} 条测试文本的 zero-shot/SFT 成对合成音频。")

    if args.dry_run:
        print("dry-run 验证通过：未加载模型，未写入评分结果。")
        return 0

    scorer: Callable[[Path, Path], float] = build_eres2net_scorer(args.model_id, args.device)
    pair_scores = score_samples(samples, enrollment_files, scorer)
    output_paths = write_results(pair_scores, args.output_dir)
    sample_rows, _ = summarize_scores(pair_scores)
    if not args.skip_plot:
        plot_path = args.output_dir / "sim_comparison.png"
        create_comparison_plot(sample_rows, plot_path)
        output_paths["plot"] = plot_path

    _print_summary(pair_scores)
    print("\nDone.")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
