from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recognition.transformer.features import (
    LANDMARK_DIM,
    MODEL_INPUT_DIM,
    SEQUENCE_LENGTH,
    featurize,
    interp_missing,
    materialize_sequence,
    resample,
)
from recognition.transformer.recognizer import (
    LABELS,
    IntegrityError,
    Knee42TransformerRecognizer,
    verify_integrity_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "realtime" / "best_current"


def _sequence(frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.4, size=(frames, LANDMARK_DIM)).astype(np.float32)


class FeatureContractTests(unittest.TestCase):
    def test_featurize_produces_the_locked_64_by_657_shape(self):
        features = featurize(_sequence(37))
        self.assertEqual(features.shape, (SEQUENCE_LENGTH, MODEL_INPUT_DIM))
        self.assertEqual(features.dtype, np.float32)

    def test_channels_are_position_then_velocity_then_acceleration(self):
        positions = resample(_sequence(80), SEQUENCE_LENGTH)
        features = featurize(_sequence(80))
        np.testing.assert_allclose(features[:, :LANDMARK_DIM], positions, rtol=0, atol=0)

        velocity = features[:, LANDMARK_DIM : 2 * LANDMARK_DIM]
        acceleration = features[:, 2 * LANDMARK_DIM :]
        np.testing.assert_allclose(velocity[1:], np.diff(positions, axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(velocity[0], 0.0, atol=0)
        np.testing.assert_allclose(acceleration[0], 0.0, atol=0)

    def test_interpolation_fills_gaps_and_holds_the_edges(self):
        values = np.zeros((5, LANDMARK_DIM), dtype=np.float32)
        values[:, 0] = [np.nan, 1.0, np.nan, 3.0, np.nan]
        filled = interp_missing(values)
        np.testing.assert_allclose(filled[:, 0], [1.0, 1.0, 2.0, 3.0, 3.0], rtol=1e-6)

    def test_a_dimension_that_is_never_observed_collapses_to_zero(self):
        values = np.zeros((4, LANDMARK_DIM), dtype=np.float32)
        values[:, 7] = np.nan
        self.assertTrue(np.all(interp_missing(values)[:, 7] == 0.0))

    def test_a_single_frame_sequence_is_repeated_not_rejected(self):
        features = materialize_sequence(_sequence(1))
        self.assertEqual(features.shape, (SEQUENCE_LENGTH, MODEL_INPUT_DIM))
        self.assertTrue(np.all(features[:, LANDMARK_DIM:] == 0.0))

    def test_wrong_landmark_width_is_rejected(self):
        with self.assertRaises(ValueError):
            interp_missing(np.zeros((8, LANDMARK_DIM - 1), dtype=np.float32))

    def test_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            interp_missing(np.zeros((0, LANDMARK_DIM), dtype=np.float32))


class BundleIntegrityTests(unittest.TestCase):
    def test_shipped_bundle_verifies(self):
        digests = verify_integrity_manifest(BUNDLE)
        self.assertIn("best_model.pt", digests)
        self.assertIn("model_card.json", digests)

    def test_a_tampered_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "bundle"
            shutil.copytree(BUNDLE, copy)
            payload = json.loads((copy / "runtime_config.json").read_text(encoding="utf-8"))
            payload["stream"] = "IR"
            (copy / "runtime_config.json").write_bytes(
                (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
            with self.assertRaises(IntegrityError):
                verify_integrity_manifest(copy)

    def test_a_missing_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "bundle"
            shutil.copytree(BUNDLE, copy)
            (copy / "model_card.json").unlink()
            with self.assertRaises(IntegrityError):
                verify_integrity_manifest(copy)


class RecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = Knee42TransformerRecognizer(BUNDLE)

    def test_label_contract_is_the_ordered_42_class_list(self):
        self.assertEqual(self.recognizer.labels, LABELS)
        self.assertEqual(len(LABELS), 42)

    def test_probabilities_are_normalized_over_42_classes(self):
        probabilities = self.recognizer.predict_proba(_sequence(50, seed=3))
        self.assertEqual(probabilities.shape, (42,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)

    def test_predict_returns_ranked_label_text_probability_triples(self):
        results = self.recognizer.predict(_sequence(50, seed=4), topk=3)
        self.assertEqual(len(results), 3)
        probabilities = [probability for _, _, probability in results]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        for label_id, display_text, _ in results:
            self.assertIn(label_id, LABELS)
            self.assertTrue(display_text.strip())

    def test_nan_inputs_are_accepted_and_match_a_pre_interpolated_run(self):
        sequence = _sequence(40, seed=5)
        with_gaps = sequence.copy()
        with_gaps[10:14, 30:60] = np.nan
        direct = self.recognizer.predict_proba(with_gaps)
        pre_filled = self.recognizer.predict_proba(interp_missing(with_gaps))
        np.testing.assert_allclose(direct, pre_filled, rtol=0, atol=0)

    def test_batch_matches_single_sequence_inference(self):
        sequences = [_sequence(30, seed=6), _sequence(45, seed=7)]
        batched = self.recognizer.predict_batch(sequences, topk=1)
        for sequence, result in zip(sequences, batched):
            self.assertEqual(result[0][0], self.recognizer.predict(sequence, topk=1)[0][0])

    def test_empty_batch_returns_empty_list(self):
        self.assertEqual(self.recognizer.predict_batch([]), [])


class PublishedMetricsTests(unittest.TestCase):
    """The README table must stay derivable from the logs committed beside it."""

    def test_the_committed_metrics_match_a_fresh_aggregation_of_the_raw_logs(self):
        import subprocess
        import sys

        expected = json.loads(
            (ROOT / "docs" / "evaluation" / "knee42_loso_metrics.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "recomputed.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "aggregate_knee42_loso_runs.py"),
                    "--runs", str(ROOT / "docs" / "evaluation" / "runs"),
                    "--out", str(out),
                ],
                check=True,
                capture_output=True,
                cwd=ROOT,
            )
            recomputed = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(recomputed, expected)

    def test_every_arm_quoted_in_the_readme_is_present_with_its_seeds(self):
        metrics = json.loads(
            (ROOT / "docs" / "evaluation" / "knee42_loso_metrics.json").read_text(encoding="utf-8")
        )
        for arm in ("transslr", "pretrained_ft", "proto", "mirror_ab", "mcc_v2"):
            self.assertIn(arm, metrics["arms"])
            summary = metrics["arms"][arm]
            self.assertGreaterEqual(summary["runs"], 12)
            for group in summary["per_test_signer"].values():
                self.assertEqual(group["seeds"], 3)


if __name__ == "__main__":
    unittest.main()
