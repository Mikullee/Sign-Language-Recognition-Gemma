from __future__ import annotations

import ast
import hashlib
import inspect
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from recognition.training.knee42_devonly import train_dev_only
from recognition.training.knee42_policy import LeakageError
from recognition.training.train_knee42_bigru import Config, train_one


ROOT = Path(__file__).resolve().parents[1]


class PublicTrainingPolicyTests(unittest.TestCase):
    def test_legacy_module_rejects_j_before_touching_features(self):
        rows = [
            {
                "sample_id": "j-only",
                "split": "test",
                "signer_id": "J",
                "label_id": "K42_01",
                "display_text": "J",
                "source": "private",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(LeakageError):
                train_one(
                    rows=rows,
                    manifest_hash="manifest",
                    split_hash="split",
                    feature_dir=root / "missing-features",
                    out_dir=root / "output",
                    seed=44,
                    device=torch.device("cpu"),
                    config=Config(),
                )

    def test_legacy_script_points_to_the_dev_only_entrypoint(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_knee42_bigru.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "train_knee42_devonly.py",
            completed.stdout + completed.stderr,
        )


class PublishedExampleTests(unittest.TestCase):
    def test_training_examples_pass_every_required_keyword(self):
        required = {
            name
            for name, parameter in inspect.signature(train_dev_only).parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        for relative_path in ("README.md", "docs/REVIEWER_GUIDE.md"):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            blocks = re.findall(r"```python\s+(.*?)```", text, flags=re.DOTALL)
            block = next(item for item in blocks if "train_dev_only(" in item)
            tree = ast.parse(block)
            call = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "train_dev_only"
            )
            supplied = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
            self.assertEqual(required - supplied, set(), relative_path)


class PublishedConfigLineageTests(unittest.TestCase):
    EXPECTED_SHA256 = {
        "round0_config.json": "2b36d53975903fd21fb0d0b9a111b61e58f14c15fd2566f97cc6d2474df0fd8f",
        "round1_config.json": "5613b9ebcabab788a4e1caae67496cb295e59d5678f63a601cf67ec3197fb5a0",
        "round2_config.json": "99a639ce264eedbeb5743be924185330586954fc7811cb6e8166073e4256980e",
        "round3_config.json": "9860bba48e139f3c08897e4a5a1f40df0c32c713aaf59d20aba0956c1d6d2d3c",
        "round4_config.json": "04609bdfec8bb796bc7c26069eab821d2f995aa3af2002ab1bd7298f120fd974",
        "round5_config.json": "5a27dbed64c4230e6dc9b59790276035a357ccace0e8741037f963a36f76e110",
        "round6_config.json": "510849ffba7ba86ff9e7cf9eef00de314b684b3f98bb1ad600da2964539f86f6",
        "round7_config.json": "041b624d67ee837c537ee5900c8d4cff5bf3f58b5db1caaa76d9c76a2fb287e6",
        "round8_config.json": "b3cdd07740bf6255e8a3d975b0ad3fccd6f8155325146dffd1da51b2a72c6e00",
        "round9_config.json": "c7470645cb971f3c33856fa27354b864cb5913a19949d08b78b2209060753262",
        "round10_config.json": "7b5ccff182c696d0a7f9fd237bc48f96c7a06c2ffd4612ff0955c9ef745d8e0a",
    }

    def test_public_configs_are_the_exact_executed_rounds(self):
        config_dir = ROOT / "configs" / "knee42"
        for name, expected_hash in self.EXPECTED_SHA256.items():
            actual_hash = hashlib.sha256((config_dir / name).read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, name)

    def test_mislabelled_proposal_configs_are_not_published_as_round_history(self):
        misleading = [
            path
            for path in (ROOT / "configs").glob("knee42_*.json")
            if path.parent == ROOT / "configs"
        ]
        self.assertEqual(misleading, [])


if __name__ == "__main__":
    unittest.main()
