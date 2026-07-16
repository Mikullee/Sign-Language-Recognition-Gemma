from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.verify_release_safety import (
    find_markers_in_stream,
    scan_portable_directory,
    scan_zip,
)


class ReleaseArtifactScannerTests(unittest.TestCase):
    def test_stream_scanner_finds_marker_across_chunk_boundary(self):
        payload = b"x" * 10 + b"tku" + b"310ai"

        found = find_markers_in_stream(io.BytesIO(payload), chunk_size=12)

        self.assertTrue(found)

    def test_clean_portable_directory_and_zip_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            portable = root / "portable"
            portable.mkdir()
            (portable / "SignLanguageRecognition.exe").write_bytes(b"offline runtime")
            zip_path = root / "portable.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("SignLanguageRecognition/readme.txt", "offline")

            self.assertEqual(scan_portable_directory(portable), [])
            self.assertEqual(scan_zip(zip_path), [])


if __name__ == "__main__":
    unittest.main()
