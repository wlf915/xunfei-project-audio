from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
import subprocess
import sys

from evaluation.run_sim import (
    DEFAULT_MODEL_ID,
    build_eres2net_scorer,
    create_comparison_plot,
    parse_args,
)


class RunSimTest(unittest.TestCase):
    def test_parses_expected_cli_defaults(self) -> None:
        args = parse_args(
            [
                "--enrollment-dir",
                "evaluation/data/enrollment",
                "--generated-dir",
                "evaluation/data/generated",
                "--output-dir",
                "evaluation/results",
            ]
        )

        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.model_id, "iic/speech_eres2netv2_sv_zh-cn_16k-common")
        self.assertFalse(args.skip_plot)

    def test_eres2net_scorer_extracts_embeddings_and_uses_cosine_similarity(self) -> None:
        class FakeAutoModel:
            def generate(self, input: str):
                embeddings = {
                    "generated.wav": [1.0, 0.0],
                    "enrollment.wav": [0.0, 1.0],
                }
                return [{"spk_embedding": embeddings[Path(input).name]}]

        captured: dict[str, str] = {}

        def fake_factory(*, model: str, device: str):
            captured["model"] = model
            captured["device"] = device
            return FakeAutoModel()

        scorer = build_eres2net_scorer(DEFAULT_MODEL_ID, "cpu", model_factory=fake_factory)
        score = scorer(Path("generated.wav"), Path("enrollment.wav"))

        self.assertEqual(captured, {"model": DEFAULT_MODEL_ID, "device": "cpu"})
        self.assertAlmostEqual(score, 0.0)

    def test_eres2net_scorer_rejects_missing_embedding(self) -> None:
        class FakeAutoModel:
            def generate(self, input: str):
                return [{"text": "not an embedding"}]

        scorer = build_eres2net_scorer(
            DEFAULT_MODEL_ID,
            "cpu",
            model_factory=lambda **_: FakeAutoModel(),
        )

        with self.assertRaisesRegex(RuntimeError, "spk_embedding"):
            scorer(Path("generated.wav"), Path("enrollment.wav"))

    def test_creates_comparison_png(self) -> None:
        rows = [
            {"system": "zero_shot", "sample_id": "test_001", "sim_mean": 0.61},
            {"system": "zero_shot", "sample_id": "test_002", "sim_mean": 0.65},
            {"system": "sft", "sample_id": "test_001", "sim_mean": 0.75},
            {"system": "sft", "sample_id": "test_002", "sim_mean": 0.78},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "sim_comparison.png"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ImportWarning)
                create_comparison_plot(rows, output)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_can_run_script_directly(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "evaluation/run_sim.py", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--enrollment-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
