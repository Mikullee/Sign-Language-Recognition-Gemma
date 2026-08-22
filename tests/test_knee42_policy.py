from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from recognition.training.knee42_policy import (
    LeakageError,
    derive_research_rows,
    validate_research_rows,
    write_research_manifest,
)


def manifest_row(split: str, signer: str, sample: str = "sample") -> dict[str, str]:
    return {
        "sample_id": f"{sample}_{split}_{signer}",
        "split": split,
        "signer_id": signer,
        "label_id": "K42_01",
        "source": "knee42",
        "video_path": f"dataset/{signer}/{sample}.mp4",
    }


class ResearchManifestTests(unittest.TestCase):
    def test_research_rows_accept_only_lpx_train_and_h_dev(self):
        rows = [
            manifest_row("train", "L"),
            manifest_row("train", "P"),
            manifest_row("train", "X"),
            manifest_row("dev", "H"),
        ]

        validate_research_rows(rows)

    def test_research_rows_reject_test_split(self):
        with self.assertRaisesRegex(LeakageError, "test"):
            validate_research_rows([manifest_row("test", "J")])

    def test_research_rows_reject_j_even_if_split_is_disguised(self):
        with self.assertRaisesRegex(LeakageError, "J"):
            validate_research_rows([manifest_row("dev", "J")])

    def test_derive_research_rows_filters_only_valid_frozen_j_test(self):
        frozen = [
            manifest_row("train", "L", "one"),
            manifest_row("dev", "H", "two"),
            manifest_row("test", "J", "three"),
        ]

        derived = derive_research_rows(frozen)

        self.assertEqual([(row["split"], row["signer_id"]) for row in derived], [("train", "L"), ("dev", "H")])

    def test_derive_research_rows_rejects_disguised_j(self):
        with self.assertRaisesRegex(LeakageError, "J"):
            derive_research_rows([manifest_row("dev", "J")])

    def test_writer_is_exclusive_and_records_source_and_derived_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "training_manifest.csv"
            destination = root / "research_manifest.csv"
            ledger = root / "research_manifest.json"
            rows = [manifest_row("train", "L"), manifest_row("dev", "H"), manifest_row("test", "J")]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            result = write_research_manifest(source, destination, ledger)

            with destination.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            recorded = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(len(written), 2)
            self.assertEqual(result["counts"], {"train": 1, "dev": 1})
            self.assertEqual(recorded["source_sha256"], result["source_sha256"])
            self.assertEqual(recorded["derived_sha256"], result["derived_sha256"])
            with self.assertRaises(FileExistsError):
                write_research_manifest(source, destination, ledger)


if __name__ == "__main__":
    unittest.main()
