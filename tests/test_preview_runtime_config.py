from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class PreviewRuntimeConfigTests(unittest.TestCase):
    """Path resolution for the runtime bundle, source tree and frozen build alike.

    The runtime-bundle *loading* tests that used to live here belonged to the
    retired 27-class daily30 app; the Transformer bundle has its own verified
    loader, covered in tests/test_knee42_transformer.py.
    """

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

    def test_environment_variables_override_every_path(self):
        from recognition.config import preview_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch.dict(
                os.environ,
                {
                    "SLR_MODELS_DIR": str(root / "m"),
                    "SLR_RUNTIME_BUNDLE_DIR": str(root / "b"),
                    "SLR_RESULTS_DIR": str(root / "r"),
                    "SLR_APP_CONFIG": str(root / "app.json"),
                },
                clear=False,
            ):
                paths = preview_paths()

        self.assertEqual(paths.models_dir, root / "m")
        self.assertEqual(paths.runtime_bundle_dir, root / "b")
        self.assertEqual(paths.results_dir, root / "r")
        self.assertEqual(paths.app_config_path, root / "app.json")

    def test_the_runtime_bundle_directory_is_the_only_bundle_path(self):
        """The legacy bundle path went with the daily30 subsystem."""
        from recognition.config import PreviewPaths

        self.assertNotIn("legacy_bundle_dir", PreviewPaths.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
