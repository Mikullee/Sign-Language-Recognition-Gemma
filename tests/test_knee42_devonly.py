from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recognition.training.knee42_devonly import (
    DevOnlyConfig,
    build_criterion,
    coordinate_affine_jitter,
    landmark_mask_dropout,
    train_dev_only,
)
from recognition.training.train_knee42_bigru import CACHE_VERSION, LABELS


ROOT = Path(__file__).resolve().parents[1]


def write_cache(path: Path, value: float) -> None:
    values = np.full((5, 219), value, dtype=np.float32)
    mask = np.ones_like(values, dtype=np.bool_)
    np.savez_compressed(path, cache_version=np.asarray(CACHE_VERSION), values=values, mask=mask)


def tiny_rows(cache_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, label in enumerate(LABELS):
        for split, signer, offset in (("train", "L", 0.0), ("dev", "H", 0.25)):
            sample_id = f"sample_{index:02d}_{split}"
            write_cache(cache_dir / f"{sample_id}.npz", index + offset)
            rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": label,
                    "display_text": label,
                    "source": "knee42",
                    "signer_id": signer,
                    "split": split,
                }
            )
    return rows


class DevOnlyTrainerTests(unittest.TestCase):
    def test_coordinate_affine_jitter_is_reproducible_and_preserves_missing_values(self):
        values = np.full((3, 219), 0.5, dtype=np.float32)
        mask = np.ones_like(values, dtype=np.bool_)
        values[1, 4] = np.nan
        mask[1, 4] = False

        first = coordinate_affine_jitter(values, mask, np.random.default_rng(17))
        second = coordinate_affine_jitter(values, mask, np.random.default_rng(17))

        np.testing.assert_allclose(first, second, equal_nan=True)
        self.assertTrue(np.isnan(first[1, 4]))
        self.assertFalse(np.allclose(first[mask], values[mask]))

    def test_coordinate_affine_jitter_can_be_disabled_by_zero_amplitudes(self):
        values = np.linspace(0.0, 1.0, 438, dtype=np.float32).reshape(2, 219)
        mask = np.ones_like(values, dtype=np.bool_)

        actual = coordinate_affine_jitter(
            values,
            mask,
            np.random.default_rng(9),
            scale_jitter=0.0,
            translation_jitter=0.0,
        )

        np.testing.assert_array_equal(actual, values)

    def test_landmark_mask_dropout_is_reproducible_and_drops_complete_xyz_points(self):
        values = np.arange(4 * 219, dtype=np.float32).reshape(4, 219)
        mask = np.ones_like(values, dtype=np.bool_)
        values[0, :3] = np.nan
        mask[0, :3] = False

        first_values, first_mask = landmark_mask_dropout(
            values,
            mask,
            np.random.default_rng(23),
            dropout_probability=0.20,
        )
        second_values, second_mask = landmark_mask_dropout(
            values,
            mask,
            np.random.default_rng(23),
            dropout_probability=0.20,
        )

        np.testing.assert_array_equal(first_mask, second_mask)
        np.testing.assert_allclose(first_values, second_values, equal_nan=True)
        point_mask = first_mask.reshape(4, 73, 3)
        self.assertTrue(np.all(point_mask == point_mask[:, :, :1]))
        self.assertTrue(np.all(~first_mask[0, :3]))
        self.assertGreater(np.count_nonzero(mask & ~first_mask), 0)
        self.assertTrue(np.all(np.isnan(first_values[~first_mask])))
        np.testing.assert_array_equal(first_values[first_mask], values[first_mask])

    def test_landmark_mask_dropout_zero_probability_is_exact_noop(self):
        values = np.linspace(0.0, 1.0, 438, dtype=np.float32).reshape(2, 219)
        mask = np.ones_like(values, dtype=np.bool_)

        actual_values, actual_mask = landmark_mask_dropout(
            values,
            mask,
            np.random.default_rng(3),
            dropout_probability=0.0,
        )

        np.testing.assert_array_equal(actual_values, values)
        np.testing.assert_array_equal(actual_mask, mask)

    def test_focal_loss_at_gamma_zero_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]], dtype=torch.float32)
        targets = torch.tensor([0, 2], dtype=torch.long)
        config = DevOnlyConfig(loss="focal", label_smoothing=0.0, focal_gamma=0.0)

        actual = build_criterion(config)(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)

        torch.testing.assert_close(actual, expected)

    def test_focal_loss_rejects_label_smoothing(self):
        with self.assertRaisesRegex(ValueError, "label_smoothing"):
            build_criterion(DevOnlyConfig(loss="focal", label_smoothing=0.15))

    def test_baseline_config_matches_frozen_seed_44_method(self):
        config = json.loads((ROOT / "configs" / "knee42_devonly_baseline.json").read_text(encoding="utf-8"))

        self.assertEqual(config["sequence_length"], 64)
        self.assertEqual(config["hidden_size"], 128)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["dropout"], 0.45)
        self.assertEqual(config["label_smoothing"], 0.08)
        self.assertEqual(config["pooling"], "mean_max")
        self.assertFalse(any("test" in key.lower() for key in config))

    def test_cli_help_is_direct_and_has_no_test_option(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_knee42_devonly.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--manifest", completed.stdout)
        self.assertIn("--feature-ledger-hash", completed.stdout)
        self.assertNotIn("--test", completed.stdout.lower())

    def test_config_schema_has_no_test_field(self):
        fields = {field.name for field in dataclasses.fields(DevOnlyConfig)}

        self.assertFalse(any("test" in name.lower() for name in fields))

    def test_one_epoch_cpu_smoke_writes_dev_outputs_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            out_dir = root / "run"
            config = DevOnlyConfig(
                sequence_length=4,
                hidden_size=2,
                num_layers=1,
                dropout=0.0,
                batch_size=42,
                epochs=1,
                patience=1,
            )

            summary = train_dev_only(
                tiny_rows(cache_dir),
                manifest_hash="manifest-hash",
                split_hash="split-hash",
                feature_ledger_hash="feature-hash",
                feature_dir=cache_dir,
                out_dir=out_dir,
                seed=44,
                device=torch.device("cpu"),
                config=config,
            )
            checkpoint = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
            saved = json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))
            names = [path.name.lower() for path in out_dir.iterdir()]

        self.assertIn("dev", summary)
        self.assertNotIn("test", summary)
        self.assertFalse(any("test" in name for name in names))
        self.assertEqual(saved["selection_metric"], "dev_macro_top1")
        self.assertEqual(checkpoint["model_config"]["num_classes"], 42)
        self.assertEqual(checkpoint["model_config"]["input_dim"], 438)
        self.assertEqual(checkpoint["manifest_sha256"], "manifest-hash")
        self.assertEqual(checkpoint["split_sha256"], "split-hash")
        self.assertEqual(checkpoint["feature_ledger_sha256"], "feature-hash")

    def test_rejects_a_j_row_before_feature_lookup(self):
        forbidden = {
            "sample_id": "missing_j_cache",
            "label_id": "K42_01",
            "display_text": "K42_01",
            "source": "knee42",
            "signer_id": "J",
            "split": "dev",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "J"):
                train_dev_only(
                    [forbidden],
                    manifest_hash="m",
                    split_hash="s",
                    feature_ledger_hash="f",
                    feature_dir=root / "missing",
                    out_dir=root / "out",
                    seed=44,
                    device=torch.device("cpu"),
                    config=DevOnlyConfig(epochs=1),
                )


if __name__ == "__main__":
    unittest.main()
