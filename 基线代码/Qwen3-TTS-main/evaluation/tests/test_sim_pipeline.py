from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.sim_pipeline import (
    GeneratedSample,
    discover_generated_audio,
    list_enrollment_audio,
    score_samples,
    summarize_scores,
    write_results,
)


class SimPipelineTest(unittest.TestCase):
    def create_wav(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        return path

    def test_discovers_matching_zero_shot_and_sft_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "generated"
            self.create_wav(root / "zero_shot" / "test_002.WAV")
            self.create_wav(root / "zero_shot" / "test_001.wav")
            self.create_wav(root / "sft" / "test_001.wav")
            self.create_wav(root / "sft" / "test_002.wav")

            samples = discover_generated_audio(root)

            self.assertEqual(
                [(item.system, item.sample_id) for item in samples],
                [
                    ("zero_shot", "test_001"),
                    ("zero_shot", "test_002"),
                    ("sft", "test_001"),
                    ("sft", "test_002"),
                ],
            )

    def test_rejects_mismatched_system_sample_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "generated"
            self.create_wav(root / "zero_shot" / "test_001.wav")
            self.create_wav(root / "sft" / "test_002.wav")

            with self.assertRaisesRegex(ValueError, "样本名不一致"):
                discover_generated_audio(root)

    def test_scores_every_generated_audio_against_all_enrollment_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enrollment_dir = root / "enrollment"
            enrollment_files = [
                self.create_wav(enrollment_dir / "real_01.wav"),
                self.create_wav(enrollment_dir / "real_02.wav"),
            ]
            samples = [
                GeneratedSample("zero_shot", "test_001", self.create_wav(root / "zero.wav")),
                GeneratedSample("sft", "test_001", self.create_wav(root / "sft.wav")),
            ]

            def fake_scorer(generated: Path, enrollment: Path) -> float:
                scores = {
                    ("zero.wav", "real_01.wav"): 0.60,
                    ("zero.wav", "real_02.wav"): 0.80,
                    ("sft.wav", "real_01.wav"): 0.70,
                    ("sft.wav", "real_02.wav"): 0.90,
                }
                return scores[(generated.name, enrollment.name)]

            pair_scores = score_samples(samples, enrollment_files, fake_scorer)
            sample_rows, summary_rows = summarize_scores(pair_scores)

            self.assertEqual(len(pair_scores), 4)
            self.assertEqual(len(sample_rows), 2)
            self.assertAlmostEqual(sample_rows[0]["sim_mean"], 0.70)
            self.assertAlmostEqual(sample_rows[1]["sim_mean"], 0.80)
            self.assertEqual(summary_rows[0]["system"], "zero_shot")
            self.assertAlmostEqual(summary_rows[1]["sim_mean"], 0.80)

    def test_writes_csv_outputs_with_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enrollment = self.create_wav(root / "enrollment.wav")
            samples = [
                GeneratedSample("zero_shot", "test_001", self.create_wav(root / "zero.wav")),
                GeneratedSample("sft", "test_001", self.create_wav(root / "sft.wav")),
            ]
            pair_scores = score_samples(samples, [enrollment], lambda *_: 0.75)

            output_paths = write_results(pair_scores, root / "results")

            self.assertTrue(output_paths["pair_scores"].is_file())
            self.assertTrue(output_paths["sample_scores"].is_file())
            self.assertTrue(output_paths["summary"].is_file())
            with output_paths["summary"].open(newline="", encoding="utf-8-sig") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(set(row), {"system", "sample_count", "sim_mean", "sim_std", "sim_min", "sim_max"})
            self.assertTrue(math.isclose(float(row["sim_mean"]), 0.75))

    def test_lists_only_wav_enrollment_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            self.create_wav(folder / "real_02.wav")
            self.create_wav(folder / "real_01.WAV")
            (folder / "note.txt").write_text("ignore", encoding="utf-8")

            files = list_enrollment_audio(folder)

            self.assertEqual([item.name for item in files], ["real_01.WAV", "real_02.wav"])


if __name__ == "__main__":
    unittest.main()
