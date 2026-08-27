from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class RealtimeEntrypointTests(unittest.TestCase):
    def test_current_module_exposes_offline_cli_with_disabled_legacy_threshold(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "recognition.realtime.realtime_infer_daily30_sentence",
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        normalized_help = " ".join(result.stdout.split())
        self.assertIn("--auto-config", result.stdout)
        self.assertIn("--end-hold-sec", result.stdout)
        self.assertIn("--app-config", result.stdout)
        self.assertIn("--min-conf-override", result.stdout)
        self.assertIn("calibration/risk-coverage evidence", normalized_help)
        self.assertNotIn("calibrated confidence", result.stdout.lower())
        self.assertNotIn("--remote-run-dir", result.stdout)
        self.assertNotIn("--start-streak-frames", result.stdout)


if __name__ == "__main__":
    unittest.main()
