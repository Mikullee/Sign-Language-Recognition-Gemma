from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from recognition.evaluation.knee42_selection import (
    REQUIRED_CANDIDATE_FILES,
    SelectionError,
    create_selection_ledger,
    sha256_file,
    verify_selection,
)
from recognition.evaluation.knee42_test_once import (
    consume_test_seal,
    evaluate_j_once,
    prepare_test_once,
    select_j_test_rows,
)
from recognition.training.knee42_devonly import DevOnlyConfig, train_dev_only
from recognition.training.train_knee42_bigru import LABELS
from tests.test_knee42_devonly import tiny_rows, write_cache


ROOT = Path(__file__).resolve().parents[1]


def make_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    for index, name in enumerate(REQUIRED_CANDIDATE_FILES):
        (candidate / name).write_bytes(f"artifact-{index}-{name}".encode("utf-8"))
    return candidate


def provenance() -> dict[str, object]:
    return {
        "run_id": "20260818_035449",
        "source_commit": "abc123",
        "source_tree_sha256": "1" * 64,
        "original_manifest_sha256": "2" * 64,
        "research_manifest_sha256": "3" * 64,
        "split_sha256": "4" * 64,
        "feature_ledger_sha256": "5" * 64,
        "seed": 44,
        "best_epoch": 11,
        "dev_evidence": {"macro_top1": 0.45, "three_seed_std": 0.01, "weak_class_score": 0.2},
        "selection_rationale": "highest Dev macro with stability guardrail",
    }


class SelectionLedgerTests(unittest.TestCase):
    def test_selection_ledger_binds_every_required_candidate_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = make_candidate(root)
            ledger = root / "selection" / "final_selection.json"

            created = create_selection_ledger(ledger, candidate, provenance())
            verified = verify_selection(ledger)

            self.assertEqual(set(created["artifacts"]), set(REQUIRED_CANDIDATE_FILES))
            self.assertEqual(verified["seed"], 44)
            self.assertEqual(verified["selection_metric"], "dev_macro_top1")
            self.assertNotIn("test", json.dumps(verified).lower())

    def test_selection_ledger_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = make_candidate(root)
            ledger = root / "selection.json"
            create_selection_ledger(ledger, candidate, provenance())

            with self.assertRaises(FileExistsError):
                create_selection_ledger(ledger, candidate, provenance())

    def test_model_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = make_candidate(root)
            ledger = root / "selection.json"
            create_selection_ledger(ledger, candidate, provenance())
            (candidate / "best_model.pt").write_bytes(b"tampered")

            with self.assertRaisesRegex(SelectionError, "SHA-256"):
                verify_selection(ledger)


class TestOnceTests(unittest.TestCase):
    def test_real_cpu_evaluation_writes_j_outputs_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            research_rows = tiny_rows(cache_dir)
            candidate = root / "candidate"
            train_dev_only(
                research_rows,
                manifest_hash="research-hash",
                split_hash="split-hash",
                feature_ledger_hash="feature-hash",
                feature_dir=cache_dir,
                out_dir=candidate,
                seed=44,
                device=torch.device("cpu"),
                config=DevOnlyConfig(sequence_length=4, hidden_size=2, num_layers=1, dropout=0.0, batch_size=42, epochs=1, patience=1),
            )
            j_rows = []
            for index, label in enumerate(LABELS):
                sample_id = f"sample_{index:02d}_j"
                write_cache(cache_dir / f"{sample_id}.npz", index + 0.5)
                j_rows.append(
                    {
                        "sample_id": sample_id,
                        "label_id": label,
                        "display_text": label,
                        "source": "knee42",
                        "signer_id": "J",
                        "split": "test",
                    }
                )
            manifest = root / "frozen_manifest.csv"
            all_rows = research_rows + j_rows
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
            selection_provenance = provenance()
            selection_provenance["original_manifest_sha256"] = sha256_file(manifest)
            ledger = root / "selection.json"
            create_selection_ledger(ledger, candidate, selection_provenance)
            out_dir = root / "j_once"

            result = evaluate_j_once(
                ledger,
                manifest,
                cache_dir,
                out_dir,
                device=torch.device("cpu"),
                expected_count=42,
            )

            self.assertEqual(result["samples"], 42)
            self.assertIn("macro_top1", result)
            self.assertTrue((out_dir / "test_metrics.json").is_file())
            self.assertTrue((out_dir / "CONSUMED.json").is_file())
            (cache_dir / "sample_00_j.npz").unlink()
            with self.assertRaisesRegex(SelectionError, "consumed"):
                evaluate_j_once(
                    ledger,
                    manifest,
                    cache_dir,
                    out_dir,
                    device=torch.device("cpu"),
                    expected_count=42,
                )

    def test_test_requires_selection_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(SelectionError, "ledger"):
                prepare_test_once(root / "test_once", root / "missing.json")

    def test_consumed_test_cannot_be_prepared_or_consumed_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = make_candidate(root)
            ledger = root / "selection.json"
            create_selection_ledger(ledger, candidate, provenance())
            test_root = root / "test_once"

            prepare_test_once(test_root, ledger)
            consume_test_seal(test_root, ledger)

            with self.assertRaisesRegex(SelectionError, "consumed"):
                prepare_test_once(test_root, ledger)
            with self.assertRaisesRegex(SelectionError, "consumed"):
                consume_test_seal(test_root, ledger)

    def test_j_selector_rejects_wrong_signer_and_count(self):
        valid = [{"sample_id": "j1", "split": "test", "signer_id": "J", "label_id": "K42_01"}]
        self.assertEqual(select_j_test_rows(valid, expected_count=1), valid)
        with self.assertRaisesRegex(SelectionError, "J-only"):
            select_j_test_rows([{**valid[0], "signer_id": "H"}], expected_count=1)
        with self.assertRaisesRegex(SelectionError, "expected 2"):
            select_j_test_rows(valid, expected_count=2)

    def test_selection_scripts_support_direct_help(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for script in ("lock_knee42_selection.py", "eval_knee42_test_once.py"):
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script), "--help"],
                    cwd=temp_dir,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
