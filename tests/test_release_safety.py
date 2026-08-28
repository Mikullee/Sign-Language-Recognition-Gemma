from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSafetyTests(unittest.TestCase):
    def test_release_source_contains_no_remote_credentials_or_personal_paths(self):
        scan_roots = [
            ROOT / "recognition",
            ROOT / "scripts",
            ROOT / "docs",
            ROOT / "README.md",
            ROOT / "requirements.txt",
            ROOT / "environment.yml",
            ROOT / "artifacts" / "realtime" / "best_current",
            ROOT / "webservice",
        ]
        texts: list[tuple[Path, str]] = []
        for scan_root in scan_roots:
            files = [scan_root] if scan_root.is_file() else list(scan_root.rglob("*"))
            for path in files:
                if path.name == "verify_release_safety.py":
                    continue
                if path.is_file() and path.suffix.lower() not in {
                    ".pt",
                    ".task",
                    ".png",
                    ".jpg",
                    ".pyc",
                }:
                    texts.append(
                        (path, path.read_text(encoding="utf-8", errors="replace"))
                    )

        banned_literals = [
            "paramiko",
            "remote-run-dir",
            "remote_run_dir",
            "SLR_REMOTE_" + "PASSWORD",
            "163.13." + "202.125",
            "tku" + "310ai",
        ]
        for path, text in texts:
            for banned in banned_literals:
                self.assertNotIn(banned, text, f"{banned!r} found in {path}")
            self.assertIsNone(
                re.search(r"[A-Za-z]:\\Users\\[^\\\r\n]+", text),
                f"personal absolute path found in {path}",
            )

    def test_runtime_bundle_is_the_42_class_transformer(self):
        """The default runtime bundle must be the current model, not a leftover one."""
        bundle = ROOT / "artifacts" / "realtime" / "best_current"
        label_map = json.loads((bundle / "label_map_knee42.json").read_text(encoding="utf-8"))
        feature = json.loads((bundle / "feature_config.json").read_text(encoding="utf-8"))
        card = json.loads((bundle / "model_card.json").read_text(encoding="utf-8"))

        labels = sorted(label_map["label_to_idx"])
        self.assertEqual(labels, [f"K42_{number:02d}" for number in range(1, 43)])
        self.assertEqual(feature["input_dim"], 657)
        self.assertEqual(feature["sequence_length"], 64)
        self.assertFalse(feature["mask_concatenated"])
        self.assertEqual(card["model_id"], "knee42-transformer-v12")

    def test_model_card_does_not_present_the_mixed_split_value_as_accuracy(self):
        """The released weights saw every signer, so they carry no held-out score."""
        card = json.loads(
            (ROOT / "artifacts" / "realtime" / "best_current" / "model_card.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIsNone(card["reported_metrics"]["shipped_checkpoint"]["held_out_score"])
        self.assertIsNone(card["training_split"]["held_out_test_set"])
        self.assertIn("Optimistic", card["training_split"]["checkpoint_reported_value"]["interpretation"])



if __name__ == "__main__":
    unittest.main()
