from __future__ import annotations

import csv
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
            ROOT / "artifacts" / "legacy" / "daily30_27class",
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

    def test_legacy_daily30_bundle_has_exactly_27_matching_labels_and_no_t09(self):
        bundle = ROOT / "artifacts" / "legacy" / "daily30_27class"
        label_map = json.loads((bundle / "label_map_v1.json").read_text(encoding="utf-8"))
        with (bundle / "fixed_sentence_templates_daily30.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            template_ids = [row["template_id"] for row in csv.DictReader(handle)]

        labels = [
            label
            for _, label in sorted(
                (int(index), label)
                for index, label in label_map["idx_to_label"].items()
            )
        ]
        self.assertEqual(len(labels), 27)
        self.assertEqual(template_ids, labels)
        self.assertNotIn("T09", labels)

    def test_auto_mode_hint_is_standing_friendly_and_has_no_space_instruction(self):
        from recognition.realtime.realtime_infer_daily30_sentence import operation_hint

        first_hint = operation_hint("auto", "IDLE_BLANK", has_result=False)
        next_hint = operation_hint("auto", "IDLE_BLANK", has_result=True)
        manual_hint = operation_hint("manual", "IDLE_BLANK", has_result=False)

        self.assertIn("站立", first_hint)
        self.assertIn("雙手自然垂放身側", first_hint)
        self.assertIn("下一句", next_hint)
        self.assertNotIn("Space", first_hint)
        self.assertIn("Space", manual_hint)

    def test_windows_packaging_files_include_offline_resources(self):
        spec_path = ROOT / "packaging" / "windows" / "SignLanguageRecognition.spec"
        build_script = ROOT / "scripts" / "build_windows_portable.ps1"
        launch_script = ROOT / "packaging" / "windows" / "start_ivcam.cmd"
        app_config = ROOT / "app_config.json"

        for path in [spec_path, build_script, launch_script, app_config]:
            self.assertTrue(path.is_file(), f"missing {path}")

        spec_text = spec_path.read_text(encoding="utf-8")
        normalized = spec_text.replace("\\", "/")
        self.assertIn("resources/models", normalized)
        self.assertIn("resources/artifacts/realtime/best_current", normalized)
        self.assertIn("collect_all", spec_text)
        self.assertIn("SignLanguageRecognition", spec_text)


if __name__ == "__main__":
    unittest.main()
