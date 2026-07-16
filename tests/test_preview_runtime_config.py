from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class PreviewRuntimeConfigTests(unittest.TestCase):
    def test_preview_paths_are_repo_relative(self):
        from recognition.config import preview_paths

        paths = preview_paths()

        self.assertEqual(paths.repo_root.name, "Sign-Language-Recognition-Gemma-preview")
        self.assertEqual(paths.models_dir, paths.repo_root / "models")
        self.assertEqual(paths.runtime_bundle_dir, paths.repo_root / "artifacts" / "realtime" / "best_current")
        self.assertEqual(paths.results_dir, paths.repo_root / "data" / "results")

    def test_load_runtime_bundle_uses_local_preview_artifacts(self):
        from recognition.inference.daily30_sentence_realtime_utils import load_runtime_bundle

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "label_map_v1.json").write_text(
                json.dumps({"idx_to_label": {"0": "T01"}, "label_to_idx": {"T01": 0}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (cache_dir / "train_summary_v1.json").write_text(
                json.dumps({"pooling": "mean_max"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (cache_dir / "launch_summary.json").write_text(
                json.dumps(
                    {
                        "run_name": "preview_bundle",
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

            bundle = load_runtime_bundle(cache_dir)

        self.assertEqual(bundle["labels"], ["T01"])
        self.assertEqual(bundle["label_display"]["T01"], "你好")
        self.assertEqual(bundle["sequence_length"], 72)
        self.assertEqual(bundle["pooling"], "mean_max")

    def test_parse_args_supports_max_frames(self):
        from recognition.realtime.realtime_infer_daily30_sentence import parse_args

        args = parse_args(["--source", "demo.mp4", "--max-frames", "120"])

        self.assertEqual(args.source, "demo.mp4")
        self.assertEqual(args.max_frames, 120)


if __name__ == "__main__":
    unittest.main()
