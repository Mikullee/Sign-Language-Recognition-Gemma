from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from recognition.training.knee42_devonly import DevOnlyConfig
from recognition.training.knee42_diagnostics import validate_diagnostic_provenance, write_diagnostics
from recognition.training.knee42_policy import LeakageError
from recognition.training.knee42_rounds import rank_candidates, validate_single_factor
from recognition.training.train_knee42_bigru import CACHE_VERSION


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticPolicyTests(unittest.TestCase):
    def test_diagnostics_accept_train_lpx_and_dev_h_predictions(self):
        rows = [
            {"sample_id": "train-l", "split": "train", "signer_id": "L"},
            {"sample_id": "dev-h", "split": "dev", "signer_id": "H"},
        ]
        predictions = [{"sample_id": "dev-h", "signer_id": "H", "true_label": "K42_01"}]

        validate_diagnostic_provenance(rows, predictions)

    def test_diagnostics_reject_j_before_any_feature_path_is_needed(self):
        rows = [{"sample_id": "j", "split": "dev", "signer_id": "J"}]

        with self.assertRaisesRegex(LeakageError, "J"):
            validate_diagnostic_provenance(rows, [])

    def test_diagnostics_reject_predictions_not_owned_by_h_dev(self):
        rows = [{"sample_id": "dev-h", "split": "dev", "signer_id": "H"}]
        predictions = [{"sample_id": "unknown", "signer_id": "J", "true_label": "K42_01"}]

        with self.assertRaisesRegex(LeakageError, "prediction"):
            validate_diagnostic_provenance(rows, predictions)

    def test_writer_records_feature_loss_weak_class_and_confusion_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_dir = root / "features"
            candidate_dir = root / "candidate"
            out_dir = root / "diagnostics"
            feature_dir.mkdir()
            candidate_dir.mkdir()
            rows = [
                {"sample_id": "train-l", "split": "train", "signer_id": "L", "label_id": "K42_01"},
                {"sample_id": "dev-h", "split": "dev", "signer_id": "H", "label_id": "K42_01"},
            ]
            for sample_id in ("train-l", "dev-h"):
                values = np.ones((3, 219), dtype=np.float32)
                mask = np.ones_like(values, dtype=np.bool_)
                values[0, 0] = np.nan
                mask[0, 0] = False
                np.savez_compressed(
                    feature_dir / f"{sample_id}.npz",
                    cache_version=np.asarray(CACHE_VERSION),
                    values=values,
                    mask=mask,
                )
            with (candidate_dir / "dev_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "signer_id", "true_label", "pred_label", "correct", "top1_probability"])
                writer.writeheader()
                writer.writerow({"sample_id": "dev-h", "signer_id": "H", "true_label": "K42_01", "pred_label": "K42_02", "correct": "0", "top1_probability": "0.4"})
            (candidate_dir / "train_history.json").write_text(
                json.dumps([{"epoch": 1, "train_loss": 2.0, "train_overall_top1": 0.5, "dev_loss": 3.0, "dev_overall_top1": 0.0, "dev_macro_top1": 0.0}]),
                encoding="utf-8",
            )

            result = write_diagnostics(rows, feature_dir, candidate_dir, out_dir)

            self.assertEqual(result["sample_counts"], {"train": 1, "dev": 1})
            self.assertEqual(result["provenance_signers"], {"train": ["L"], "dev": ["H"]})
            self.assertGreater(result["features"]["overall_missing_rate"], 0.0)
            self.assertEqual(result["weak_classes"][0]["label_id"], "K42_01")
            self.assertEqual(result["confusion_pairs"][0]["pred_label"], "K42_02")
            self.assertTrue((out_dir / "diagnostics.json").is_file())
            self.assertTrue((out_dir / "loss_curve.png").is_file())


class RoundPolicyTests(unittest.TestCase):
    @staticmethod
    def _round(number: int) -> DevOnlyConfig:
        path = ROOT / "configs" / "knee42" / f"round{number}_config.json"
        return DevOnlyConfig(**json.loads(path.read_text()))

    def test_round1_adds_coordinate_jitter_to_round0(self):
        changed = validate_single_factor(self._round(0), self._round(1), "augmentation")
        self.assertEqual(changed, {"augmentation"})
        self.assertEqual(self._round(1).augmentation, "coordinate_jitter")

    def test_round2_changes_only_focal_loss_from_round1(self):
        changed = validate_single_factor(self._round(1), self._round(2), "loss")
        self.assertEqual(changed, {"loss", "label_smoothing"})
        self.assertEqual(self._round(2).loss, "focal")

    def test_round3_changes_only_sequence_length_from_round1(self):
        changed = validate_single_factor(self._round(1), self._round(3), "temporal")
        self.assertEqual(changed, {"sequence_length"})
        self.assertEqual(self._round(3).sequence_length, 96)

    def test_round4_adds_landmark_dropout_to_round3(self):
        changed = validate_single_factor(self._round(3), self._round(4), "augmentation")
        self.assertEqual(changed, {"augmentation", "landmark_dropout_probability"})
        self.assertEqual(self._round(4).landmark_dropout_probability, 0.05)

    def test_round5_changes_only_dropout_from_round3(self):
        changed = validate_single_factor(self._round(3), self._round(5), "architecture")
        self.assertEqual(changed, {"dropout"})
        self.assertEqual(self._round(5).dropout, 0.60)

    def test_round6_changes_only_weight_decay_from_round3(self):
        changed = validate_single_factor(self._round(3), self._round(6), "optimization")
        self.assertEqual(changed, {"weight_decay"})
        self.assertEqual(self._round(6).weight_decay, 0.001)

    def test_round7_changes_only_pooling_from_round3(self):
        changed = validate_single_factor(self._round(3), self._round(7), "architecture")
        self.assertEqual(changed, {"pooling"})
        self.assertEqual(self._round(7).pooling, "mean")

    def test_round8_changes_only_hidden_size_from_round3(self):
        changed = validate_single_factor(self._round(3), self._round(8), "architecture")
        self.assertEqual(changed, {"hidden_size"})
        self.assertEqual(self._round(8).hidden_size, 192)

    def test_round9_changes_only_recurrent_depth_from_round3(self):
        changed = validate_single_factor(self._round(3), self._round(9), "architecture")
        self.assertEqual(changed, {"num_layers"})
        self.assertEqual(self._round(9).num_layers, 1)

    def test_round10_changes_only_sequence_length_from_round1(self):
        changed = validate_single_factor(self._round(1), self._round(10), "temporal")
        self.assertEqual(changed, {"sequence_length"})
        self.assertEqual(self._round(10).sequence_length, 128)

    def test_diagnostic_and_round_scripts_support_direct_help(self):
        for script in ("diagnose_knee42_dev.py", "run_knee42_round.py"):
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_round_accepts_exactly_one_declared_factor_group(self):
        baseline = DevOnlyConfig()
        candidate = replace(baseline, label_smoothing=0.0, loss="focal")

        changed = validate_single_factor(baseline, candidate, "loss")

        self.assertEqual(changed, {"label_smoothing", "loss"})

    def test_round_rejects_changes_across_two_factor_groups(self):
        baseline = DevOnlyConfig()
        candidate = replace(baseline, label_smoothing=0.0, hidden_size=192)

        with self.assertRaisesRegex(ValueError, "one factor group"):
            validate_single_factor(baseline, candidate, "loss")

    def test_round_rejects_no_change(self):
        baseline = DevOnlyConfig()

        with self.assertRaisesRegex(ValueError, "no configuration change"):
            validate_single_factor(baseline, baseline, "loss")

    def test_candidate_ranking_uses_dev_macro_then_stability_then_weak_classes(self):
        candidates = [
            {"name": "less-stable", "dev_macro_top1": 0.45, "dev_macro_std": 0.03, "weak_class_score": 0.20},
            {"name": "winner", "dev_macro_top1": 0.45, "dev_macro_std": 0.01, "weak_class_score": 0.18},
            {"name": "lower", "dev_macro_top1": 0.44, "dev_macro_std": 0.00, "weak_class_score": 0.50},
        ]

        ranked = rank_candidates(candidates)

        self.assertEqual([item["name"] for item in ranked], ["winner", "less-stable", "lower"])

    def test_candidate_ranking_rejects_test_metrics_anywhere(self):
        with self.assertRaisesRegex(LeakageError, "Test"):
            rank_candidates([{"name": "bad", "dev_macro_top1": 0.9, "test": {"macro_top1": 1.0}}])


if __name__ == "__main__":
    unittest.main()
