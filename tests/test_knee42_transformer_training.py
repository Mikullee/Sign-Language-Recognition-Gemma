from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recognition.transformer.features import LANDMARK_DIM, SEQUENCE_LENGTH, featurize
from recognition.transformer.model import Knee42Transformer
from recognition.training.knee42_transformer import (
    CACHE_VERSION,
    Dataset,
    augment,
    evaluate,
    load_dataset,
    save_checkpoint,
    train_final,
    train_leave_one_signer_out,
)


LABELS = [f"K42_{number:02d}" for number in range(1, 4)]
SIGNERS = ["H", "L", "P"]


def _write_cache(root: Path, *, cache_version: str = CACHE_VERSION, width: int = LANDMARK_DIM):
    """A tiny stand-in for the real feature cache: 3 classes, 3 signers, 18 samples."""
    features = root / "features_final"
    features.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for label_index, label in enumerate(LABELS):
        for signer in SIGNERS:
            for trial in range(2):
                sample_id = f"{label}_{signer}_{trial}"
                frames = 20 + trial * 5
                values = rng.normal(label_index, 0.05, size=(frames, width)).astype(np.float32)
                values[0, 0] = np.nan
                np.savez(
                    features / f"{sample_id}.npz",
                    cache_version=np.asarray(cache_version),
                    values=values,
                    mask=np.isfinite(values),
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label_id": label,
                        "signer_id": signer,
                        "split": "train",
                    }
                )
    with (root / "research_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


class DatasetTests(unittest.TestCase):
    def test_a_cache_loads_with_labels_signers_and_no_remaining_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_dataset(_write_cache(Path(directory)))
        self.assertEqual(len(dataset), 18)
        self.assertEqual(dataset.label_ids, LABELS)
        self.assertEqual(sorted(set(dataset.signers)), SIGNERS)
        self.assertTrue(all(np.isfinite(sequence).all() for sequence in dataset.sequences))

    def test_a_wrong_cache_version_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_cache(Path(directory), cache_version="something_else")
            with self.assertRaises(ValueError) as caught:
                load_dataset(root)
        self.assertIn("cache_version", str(caught.exception))

    def test_a_wrong_landmark_width_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_cache(Path(directory), width=LANDMARK_DIM - 1)
            with self.assertRaises(ValueError):
                load_dataset(root)

    def test_a_missing_manifest_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_dataset(Path(directory))


class AugmentationTests(unittest.TestCase):
    def test_augmentation_keeps_the_landmark_width_and_stays_usable(self):
        rng = np.random.RandomState(0)
        sequence = np.random.default_rng(1).normal(size=(40, LANDMARK_DIM)).astype(np.float32)
        for _ in range(20):
            augmented = augment(sequence, rng)
            self.assertEqual(augmented.shape[1], LANDMARK_DIM)
            self.assertGreaterEqual(len(augmented), 8)
            self.assertEqual(featurize(augmented).shape[0], SEQUENCE_LENGTH)

    def test_augmentation_actually_changes_the_sequence(self):
        rng = np.random.RandomState(3)
        sequence = np.random.default_rng(2).normal(size=(40, LANDMARK_DIM)).astype(np.float32)
        self.assertFalse(np.array_equal(augment(sequence, rng), sequence))


class EvaluateTests(unittest.TestCase):
    def test_a_perfect_model_scores_one_across_every_metric(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        features = np.zeros((len(labels), SEQUENCE_LENGTH, LANDMARK_DIM * 3), dtype=np.float32)

        class _Perfect(torch.nn.Module):
            def forward(self, batch):
                out = torch.full((len(batch), 3), -10.0)
                for row, label in enumerate(labels[: len(batch)]):
                    out[row, label] = 10.0
                return out

        scores = evaluate(_Perfect(), features, labels, "cpu")
        self.assertEqual(scores["top1"], 1.0)
        self.assertEqual(scores["macro_top1"], 1.0)

    def test_macro_weights_classes_equally_not_samples(self):
        labels = np.asarray([0] * 9 + [1])
        logits = np.zeros((10, 2), dtype=np.float32)
        logits[:, 0] = 1.0  # always predicts class 0

        class _Always(torch.nn.Module):
            def forward(self, batch):
                return torch.from_numpy(logits[: len(batch)])

        features = np.zeros((10, SEQUENCE_LENGTH, LANDMARK_DIM * 3), dtype=np.float32)
        scores = evaluate(_Always(), features, labels, "cpu")
        self.assertAlmostEqual(scores["top1"], 0.9)
        self.assertAlmostEqual(scores["macro_top1"], 0.5)


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.dataset = load_dataset(_write_cache(Path(cls._directory.name)))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_leave_one_signer_out_holds_that_signer_back(self):
        _, metrics = train_leave_one_signer_out(
            self.dataset, "H", seed=0, device="cpu", epochs=1, patience=1, batch_size=4
        )
        self.assertEqual(metrics["protocol"], "leave_one_signer_out")
        self.assertEqual(metrics["test_signer"], "H")
        self.assertIn("macro_top1", metrics["test"])

    def test_an_unknown_signer_is_rejected(self):
        with self.assertRaises(ValueError):
            train_leave_one_signer_out(self.dataset, "Z", seed=0, device="cpu", epochs=1)

    def test_final_training_reports_no_held_out_set_and_warns(self):
        _, metrics = train_final(
            self.dataset, seed=0, device="cpu", epochs=1, patience=1, batch_size=4
        )
        self.assertEqual(metrics["protocol"], "all_signers")
        self.assertIsNone(metrics["held_out_test_set"])
        self.assertIn("optimistic", metrics["warning"])
        self.assertNotIn("test", metrics)

    def test_a_saved_checkpoint_reloads_into_the_shipped_model_class(self):
        model, metrics = train_final(
            self.dataset, seed=0, device="cpu", epochs=1, patience=1, batch_size=4
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(model, self.dataset, path, metrics)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["n_classes"], len(LABELS))
            self.assertEqual(payload["label_ids"], LABELS)
            reloaded = Knee42Transformer(payload["n_classes"])
            reloaded.load_state_dict(payload["state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
