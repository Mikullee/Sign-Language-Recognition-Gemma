from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PreviewRuntimeConfigTests(unittest.TestCase):
    def test_preview_paths_are_repo_relative(self):
        from recognition.config import preview_paths

        paths = preview_paths()

        self.assertTrue((paths.repo_root / "recognition").is_dir())
        self.assertEqual(paths.models_dir, paths.repo_root / "models")
        self.assertEqual(
            paths.runtime_bundle_dir,
            paths.repo_root / "artifacts" / "realtime" / "best_current",
        )
        self.assertEqual(paths.results_dir, paths.repo_root / "data" / "results")
        self.assertEqual(paths.app_config_path, paths.repo_root / "app_config.json")

    def test_frozen_paths_read_resources_and_write_logs_next_to_exe(self):
        from recognition.config import preview_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_root = root / "_internal"
            exe_path = root / "portable" / "SignLanguageRecognition.exe"
            with (
                mock.patch.object(sys, "frozen", True, create=True),
                mock.patch.object(sys, "_MEIPASS", str(bundle_root), create=True),
                mock.patch.object(sys, "executable", str(exe_path)),
                mock.patch.dict(
                    os.environ,
                    {
                        "SLR_MODELS_DIR": "",
                        "SLR_RUNTIME_BUNDLE_DIR": "",
                        "SLR_RESULTS_DIR": "",
                        "SLR_APP_CONFIG": "",
                    },
                    clear=False,
                ),
            ):
                paths = preview_paths()

        self.assertEqual(paths.models_dir, bundle_root / "resources" / "models")
        self.assertEqual(
            paths.runtime_bundle_dir,
            bundle_root / "resources" / "artifacts" / "realtime" / "best_current",
        )
        self.assertEqual(paths.results_dir, exe_path.parent / "logs")
        self.assertEqual(paths.app_config_path, exe_path.parent / "app_config.json")

    def test_load_runtime_bundle_uses_local_artifacts(self):
        from recognition.inference.daily30_sentence_realtime_utils import load_runtime_bundle

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self._write_bundle(cache_dir)
            bundle = load_runtime_bundle(cache_dir)

        self.assertEqual(bundle["labels"], ["T01"])
        self.assertEqual(bundle["label_display"]["T01"], "你好")
        self.assertEqual(bundle["sequence_length"], 72)
        self.assertEqual(bundle["pooling"], "mean_max")

    def test_runtime_bundle_rejects_template_label_mismatch(self):
        from recognition.inference.daily30_sentence_realtime_utils import load_runtime_bundle

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self._write_bundle(cache_dir)
            (cache_dir / "fixed_sentence_templates_daily30.csv").write_text(
                "template_id,sentence_text\nT01,你好\nT09,我聽不懂\n",
                encoding="utf-8-sig",
            )
            with self.assertRaisesRegex(ValueError, "template"):
                load_runtime_bundle(cache_dir)

    def test_missing_runtime_bundle_fails_without_remote_fetch(self):
        from recognition.inference.daily30_sentence_realtime_utils import ensure_artifacts_cached

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "offline runtime bundle"):
                ensure_artifacts_cached(Path(tmp))

    def test_parse_args_supports_max_frames(self):
        from recognition.realtime.realtime_infer_daily30_sentence import (
            parse_args,
            resolve_runtime_args,
        )

        args = resolve_runtime_args(
            parse_args(["--source", "demo.mp4", "--max-frames", "120"])
        )

        self.assertEqual(args.source, "demo.mp4")
        self.assertEqual(args.max_frames, 120)

    def test_app_config_loads_defaults_and_cli_values_win(self):
        from recognition.realtime.realtime_infer_daily30_sentence import (
            parse_args,
            resolve_runtime_args,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "app_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "source": "2",
                        "backend": "auto",
                        "trigger_mode": "manual",
                        "save_log": False,
                    }
                ),
                encoding="utf-8",
            )
            args = resolve_runtime_args(
                parse_args(
                    [
                        "--app-config",
                        str(config_path),
                        "--source",
                        "1",
                        "--trigger-mode",
                        "auto",
                        "--save-log",
                    ]
                )
            )

        self.assertEqual(args.source, "1")
        self.assertEqual(args.backend, "auto")
        self.assertEqual(args.trigger_mode, "auto")
        self.assertTrue(args.save_log)

    def test_auto_config_loads_json_and_cli_values_win(self):
        from recognition.realtime.realtime_infer_daily30_sentence import (
            build_auto_trigger_config,
            parse_args,
            resolve_runtime_args,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "best_auto_trigger.json"
            config_path.write_text(
                json.dumps(
                    {
                        "end_hold_sec": 0.60,
                        "end_rest_vote_ratio": 0.90,
                        "hidden_rest_enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            args = resolve_runtime_args(
                parse_args(
                    [
                        "--auto-config",
                        str(config_path),
                        "--end-hold-sec",
                        "0.40",
                        "--no-hidden-rest-enabled",
                    ]
                )
            )
            config = build_auto_trigger_config(args)

        self.assertEqual(config.end_hold_sec, 0.40)
        self.assertEqual(config.end_rest_vote_ratio, 0.90)
        self.assertFalse(config.hidden_rest_enabled)

    def test_auto_config_defaults_to_visible_rest_only(self):
        from recognition.realtime.realtime_infer_daily30_sentence import (
            build_auto_trigger_config,
            parse_args,
            resolve_runtime_args,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = build_auto_trigger_config(
                resolve_runtime_args(parse_args(["--model-cache-dir", tmp]))
            )

        self.assertFalse(config.hidden_rest_enabled)
        self.assertEqual(config.end_hold_sec, 0.50)

    def test_auto_config_uses_best_config_from_model_bundle_when_present(self):
        from recognition.realtime.realtime_infer_daily30_sentence import (
            build_auto_trigger_config,
            parse_args,
            resolve_runtime_args,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            (bundle_dir / "best_auto_trigger.json").write_text(
                json.dumps({"end_hold_sec": 0.60, "end_rest_vote_ratio": 0.90}),
                encoding="utf-8",
            )
            args = resolve_runtime_args(
                parse_args(["--model-cache-dir", str(bundle_dir)])
            )
            config = build_auto_trigger_config(args)

        self.assertEqual(config.end_hold_sec, 0.60)
        self.assertEqual(config.end_rest_vote_ratio, 0.90)

    @staticmethod
    def _write_bundle(cache_dir: Path) -> None:
        (cache_dir / "label_map_v1.json").write_text(
            json.dumps(
                {"idx_to_label": {"0": "T01"}, "label_to_idx": {"T01": 0}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (cache_dir / "train_summary_v1.json").write_text(
            json.dumps({"pooling": "mean_max"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (cache_dir / "launch_summary.json").write_text(
            json.dumps(
                {
                    "run_name": "local_bundle",
                    "sequence_length": 72,
                    "frame_step": 1,
                    "hidden_size": 160,
                    "num_layers": 2,
                    "dropout": 0.55,
                    "pooling": "mean_max",
                    "append_delta": True,
                    "zscore_features": True,
                    "device": "auto",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (cache_dir / "fixed_sentence_templates_daily30.csv").write_text(
            "template_id,sentence_text\nT01,你好\n",
            encoding="utf-8-sig",
        )
        (cache_dir / "best_model.pt").write_bytes(b"model")
        (cache_dir / "best_auto_trigger.json").write_text(
            json.dumps({"end_hold_sec": 0.50}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
