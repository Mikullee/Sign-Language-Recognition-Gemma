from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from recognition.realtime.knee42_integrity import IntegrityError, VerifiedRelease
from recognition.realtime.knee42_ivcam import load_bundle


REPO = Path(__file__).resolve().parents[1]
GOLDEN_MODULE = REPO / "recognition" / "realtime" / "knee42_golden.py"
GENERATOR_MODULE = REPO / "tests" / "fixtures" / "knee42_tiny_component.py"
TINY_COMPONENT_MANIFEST_SHA256 = (
    "3dc5b76124a063f228f8e7309840d1c395b9ecaa8556d25e8ae5edc628c535be"
)
TRACKED_COMPONENT_DIR = REPO / "tests" / "fixtures" / "knee42_model_component"
TRACKED_COMPONENT_SIDECAR = (
    REPO
    / "tests"
    / "fixtures"
    / "knee42_model_component.expected-manifest-sha256.txt"
)
TRACKED_GOLDEN_CONTRACT = REPO / "tests" / "fixtures" / "knee42_golden_contract.json"
TINY_GOLDEN_TENSOR_SHA256 = (
    "279b63de8282cd8e9ae92798e04abcf55bc7caec76dae64a3993be475232a15a"
)
TINY_GOLDEN_LOGITS_SHA256 = (
    "3a344128c28719afac6400e45e278471c3b8b848c196a6c60ea70bac47b763e7"
)
TINY_MODEL_SHA256 = (
    "358cf3a493225ba27a5977bb4ccf0dda5d4ee434b9cebec8ee1b205771b9421a"
)


def import_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trusted_release(root: Path, *, golden_hash: str) -> VerifiedRelease:
    component_path = root / "model" / "component_manifest.json"
    component = json.loads(component_path.read_text(encoding="utf-8"))
    file_hashes = {
        f"model/{path.name}": sha256(path)
        for path in (root / "model").iterdir()
        if path.is_file()
    }
    file_hashes["golden_contract.json"] = golden_hash
    authenticated_files = {
        f"model/{path.name}": path.read_bytes()
        for path in (root / "model").iterdir()
        if path.is_file()
    }
    golden_path = root / "golden_contract.json"
    if golden_path.is_file():
        authenticated_files["golden_contract.json"] = golden_path.read_bytes()
    return VerifiedRelease(
        root=root.resolve(),
        release_version="v1.0.1-v13.1",
        app_version="v13.1",
        component_id=component["component_id"],
        model_version=component["model_version"],
        model_component_manifest_sha256=sha256(component_path),
        label_count=42,
        input_shape=(1, 64, 438),
        source_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        root_manifest_sha256="c" * 64,
        file_hashes=file_hashes,
        authenticated_files=authenticated_files,
    )


class TinyComponentTests(unittest.TestCase):
    def require_generator(self):
        self.assertTrue(GENERATOR_MODULE.is_file(), "tiny component generator is missing")
        module = import_file(GENERATOR_MODULE, "knee42_tiny_component_fixture")
        self.assertTrue(callable(getattr(module, "generate_tiny_component", None)))
        return module

    def test_tiny_component_is_byte_reproducible_and_uses_only_sentinel_tasks(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            first = parent / "first"
            second = parent / "second"
            first.mkdir()
            second.mkdir()
            generator.generate_tiny_component(first)
            generator.generate_tiny_component(second)

            first_hashes = {path.name: sha256(path) for path in first.iterdir()}
            second_hashes = {path.name: sha256(path) for path in second.iterdir()}
            first_size = sum(path.stat().st_size for path in first.iterdir())
            hand_bytes = (first / "hand_landmarker.task").read_bytes()
            pose_bytes = (first / "pose_landmarker.task").read_bytes()

        self.assertEqual(first_hashes, second_hashes)
        self.assertIn("component_manifest.json", first_hashes)
        self.assertIn("integrity_manifest.sha256", first_hashes)
        self.assertLess(first_size, 1_000_000)
        self.assertEqual(hand_bytes, b"TEST-SENTINEL-HAND-NOT-MEDIAPIPE\n")
        self.assertEqual(pose_bytes, b"TEST-SENTINEL-POSE-NOT-MEDIAPIPE\n")

    def test_tiny_component_generator_rejects_stale_extra_paths_before_writing(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            stale = model_dir / "stale-private.bin"
            stale.write_bytes(b"must not be packaged")

            with self.assertRaisesRegex(ValueError, "unexpected.*stale-private.bin"):
                generator.generate_tiny_component(model_dir)

            self.assertEqual(
                {path.name for path in model_dir.iterdir()},
                {"stale-private.bin"},
            )

    def test_tiny_component_runs_real_cpu_model_forward_with_exactly_42_logits(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            generator.generate_tiny_component(model_dir)
            trust = trusted_release(Path(temp_dir), golden_hash="d" * 64)
            bundle = load_bundle(
                model_dir,
                device=torch.device("cpu"),
                trusted_release=trust,
            )
            prepared = np.zeros((64, 438), dtype=np.float32)
            logits = bundle.forward_prepared(prepared)

        self.assertEqual(list(logits.shape), [1, 42])
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(bundle.component_id, "knee42-v42-tiny-test")
        self.assertEqual(bundle.model_display_version, "v42")

    def test_bundle_deserializes_only_authenticated_snapshot_and_keeps_task_bytes(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            generator.generate_tiny_component(model_dir)
            trust = trusted_release(root, golden_hash="d" * 64)
            expected_hand = trust.authenticated_files[
                "model/hand_landmarker.task"
            ]
            expected_pose = trust.authenticated_files[
                "model/pose_landmarker.task"
            ]
            for path in model_dir.iterdir():
                if path.is_file():
                    path.write_bytes(b"SWAPPED-AFTER-ROOT-VERIFICATION")

            bundle = load_bundle(
                model_dir,
                device=torch.device("cpu"),
                trusted_release=trust,
            )
            logits = bundle.forward_prepared(
                np.zeros((64, 438), dtype=np.float32)
            )

        self.assertEqual(list(logits.shape), [1, 42])
        self.assertEqual(bundle.hand_landmarker_task_bytes, expected_hand)
        self.assertEqual(bundle.pose_landmarker_task_bytes, expected_pose)

    def test_generator_has_cwd_independent_external_component_anchor(self):
        self.require_generator()
        script = f"""
import hashlib
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, {str(REPO)!r})
spec = importlib.util.spec_from_file_location('tiny_component_external', {str(GENERATOR_MODULE)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target = Path(sys.argv[1])
target.mkdir(parents=True)
module.generate_tiny_component(target)
print(hashlib.sha256((target / 'component_manifest.json').read_bytes()).hexdigest())
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            anchors = []
            for name in ("cwd-a", "cwd-b"):
                cwd = root / name
                cwd.mkdir()
                output = root / f"output-{name}"
                completed = subprocess.run(
                    [sys.executable, "-B", "-c", script, str(output)],
                    cwd=cwd,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                anchors.append(completed.stdout.strip())

        self.assertEqual(anchors[0], anchors[1])
        self.assertEqual(anchors[0], TINY_COMPONENT_MANIFEST_SHA256)

    def test_tracked_component_and_sidecar_are_exact_generator_output(self):
        generator = self.require_generator()
        expected_names = {
            "best_model.pt",
            "runtime_config.json",
            "feature_config.json",
            "standardizer_train_only.npz",
            "label_map_knee42.json",
            "display_text_map.json",
            "selection_ledger.json",
            "hand_landmarker.task",
            "pose_landmarker.task",
            "integrity_manifest.sha256",
            "component_manifest.json",
        }
        self.assertTrue(TRACKED_COMPONENT_DIR.is_dir())
        self.assertTrue(TRACKED_COMPONENT_SIDECAR.is_file())
        self.assertEqual(
            {path.name for path in TRACKED_COMPONENT_DIR.iterdir() if path.is_file()},
            expected_names,
        )
        sidecar = TRACKED_COMPONENT_SIDECAR.read_text(encoding="ascii")
        self.assertEqual(sidecar, TINY_COMPONENT_MANIFEST_SHA256 + "\n")
        self.assertEqual(
            sha256(TRACKED_COMPONENT_DIR / "component_manifest.json"),
            TINY_COMPONENT_MANIFEST_SHA256,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            regenerated = Path(temp_dir) / "model"
            regenerated.mkdir()
            generator.generate_tiny_component(regenerated)
            for name in sorted(expected_names):
                with self.subTest(name=name):
                    self.assertEqual(
                        (regenerated / name).read_bytes(),
                        (TRACKED_COMPONENT_DIR / name).read_bytes(),
                    )

    def test_tiny_fixture_binary_paths_are_trackable_with_stable_attributes(self):
        binary_names = {
            "best_model.pt",
            "standardizer_train_only.npz",
            "hand_landmarker.task",
            "pose_landmarker.task",
        }
        text_names = {
            "component_manifest.json",
            "display_text_map.json",
            "feature_config.json",
            "integrity_manifest.sha256",
            "label_map_knee42.json",
            "runtime_config.json",
            "selection_ledger.json",
        }
        for name in sorted(binary_names):
            relative = (TRACKED_COMPONENT_DIR / name).relative_to(REPO).as_posix()
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--", relative],
                cwd=REPO,
                check=False,
            )
            with self.subTest(name=name, contract="not ignored"):
                self.assertEqual(ignored.returncode, 1)
            attributes = subprocess.run(
                ["git", "check-attr", "text", "--", relative],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
            with self.subTest(name=name, contract="binary"):
                self.assertEqual(attributes.returncode, 0, attributes.stderr)
                self.assertTrue(attributes.stdout.rstrip().endswith("text: unset"))
        for name in sorted(text_names):
            relative = (TRACKED_COMPONENT_DIR / name).relative_to(REPO).as_posix()
            attributes = subprocess.run(
                ["git", "check-attr", "text", "eol", "--", relative],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
            with self.subTest(name=name, contract="LF text"):
                self.assertEqual(attributes.returncode, 0, attributes.stderr)
                self.assertIn("text: set", attributes.stdout)
                self.assertIn("eol: lf", attributes.stdout)

    def test_generator_emits_canonical_lf_text_and_stored_npz(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            generator.generate_tiny_component(model_dir)
            for path in model_dir.iterdir():
                if path.suffix == ".json" or path.name.endswith(".sha256"):
                    with self.subTest(name=path.name):
                        raw_bytes = path.read_bytes()
                        self.assertNotIn(b"\r\n", raw_bytes)
                        self.assertTrue(raw_bytes.endswith(b"\n"))
            with zipfile.ZipFile(
                model_dir / "standardizer_train_only.npz",
                "r",
            ) as archive:
                compression_types = {
                    item.compress_type for item in archive.infolist()
                }

        self.assertEqual(compression_types, {zipfile.ZIP_STORED})

    def test_tracked_golden_is_a_fixed_reviewed_cpu_software_contract(self):
        self.assertTrue(
            TRACKED_GOLDEN_CONTRACT.is_file(),
            "tracked root-level tiny golden contract is missing",
        )
        if not TRACKED_GOLDEN_CONTRACT.is_file():
            return
        raw_bytes = TRACKED_GOLDEN_CONTRACT.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        self.assertNotIn(b"\r\n", raw_bytes)
        self.assertTrue(raw_bytes.endswith(b"\n"))
        self.assertEqual(payload["component_id"], "knee42-v42-tiny-test")
        self.assertEqual(payload["model_version"], "v42")
        self.assertEqual(payload["model_sha256"], TINY_MODEL_SHA256)
        self.assertEqual(payload["tensor_sha256"], TINY_GOLDEN_TENSOR_SHA256)
        self.assertEqual(payload["logits_sha256"], TINY_GOLDEN_LOGITS_SHA256)
        self.assertEqual(payload["purpose"], "software_contract_not_accuracy_evidence")
        self.assertEqual(payload["tensor_shape"], [64, 438])
        self.assertEqual(payload["logits_shape"], [1, 42])

    def test_golden_regenerator_is_byte_identical_to_fixed_expected_fixture(self):
        generator = self.require_generator()
        regenerator = getattr(generator, "generate_tiny_golden_contract", None)
        self.assertTrue(callable(regenerator), "tiny golden regenerator is missing")
        if not callable(regenerator):
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            generator.generate_tiny_component(model_dir)
            generated = root / "golden_contract.json"
            regenerator(model_dir, generated)

            self.assertEqual(
                generated.read_bytes(),
                TRACKED_GOLDEN_CONTRACT.read_bytes(),
            )

    def test_bundle_rejects_forged_verified_payload_before_deserializing_model(self):
        generator = self.require_generator()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            generator.generate_tiny_component(model_dir)
            trust = trusted_release(root, golden_hash="d" * 64)
            forged_hashes = dict(trust.file_hashes)
            forged_hashes["model/runtime_config.json"] = "f" * 64
            forged = dataclasses.replace(trust, file_hashes=forged_hashes)
            with mock.patch(
                "recognition.realtime.knee42_ivcam.torch.load",
                side_effect=AssertionError("model deserialized before verified payload binding"),
            ) as loader:
                with self.assertRaisesRegex(IntegrityError, "trusted release.*runtime_config"):
                    load_bundle(
                        model_dir,
                        device=torch.device("cpu"),
                        trusted_release=forged,
                    )
            loader.assert_not_called()


class GoldenContractTests(unittest.TestCase):
    def require_modules(self):
        self.assertTrue(GOLDEN_MODULE.is_file(), "golden contract module is missing")
        self.assertTrue(GENERATOR_MODULE.is_file(), "tiny component generator is missing")
        golden = import_file(GOLDEN_MODULE, "knee42_golden_contract")
        generator = import_file(GENERATOR_MODULE, "knee42_tiny_component_fixture_golden")
        for name in (
            "materialize_golden_tensor",
            "golden_contract_payload",
            "load_golden_contract",
            "verify_golden_result",
        ):
            self.assertTrue(callable(getattr(golden, name, None)), name)
        return golden, generator

    def make_contract(self, root: Path):
        golden, generator = self.require_modules()
        model_dir = root / "model"
        model_dir.mkdir()
        generator.generate_tiny_component(model_dir)
        provisional_trust = trusted_release(root, golden_hash="d" * 64)
        bundle = load_bundle(
            model_dir,
            device=torch.device("cpu"),
            trusted_release=provisional_trust,
        )
        tensor = golden.materialize_golden_tensor(bundle.mean, bundle.std)
        logits = bundle.forward_prepared(tensor)
        provisional = trusted_release(root, golden_hash="d" * 64)
        payload = golden.golden_contract_payload(provisional, tensor, logits)
        path = root / "golden_contract.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        trusted = trusted_release(root, golden_hash=sha256(path))
        return golden, bundle, tensor, logits, path, trusted

    def test_golden_recipe_covers_asymmetry_missing_mask_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden, _bundle, tensor, logits, path, trusted = self.make_contract(Path(temp_dir))
            contract = golden.load_golden_contract(path, trusted_release=trusted)
            golden.verify_golden_result(contract, tensor, logits)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["recipe"], "knee42_asymmetric_missing_mask_v1")
        self.assertEqual(payload["purpose"], "software_contract_not_accuracy_evidence")
        self.assertEqual(payload["tensor_dtype"], "<f4")
        self.assertEqual(payload["tensor_shape"], [64, 438])
        self.assertEqual(payload["logits_dtype"], "<f4")
        self.assertEqual(payload["logits_shape"], [1, 42])
        self.assertEqual(
            payload["tensor_sha256"],
            hashlib.sha256(np.asarray(tensor, dtype="<f4").tobytes(order="C")).hexdigest(),
        )
        self.assertEqual(
            payload["logits_sha256"],
            hashlib.sha256(
                np.asarray(logits.detach().cpu().numpy(), dtype="<f4").tobytes(order="C")
            ).hexdigest(),
        )
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue(np.any(tensor[:, 219:] == 0.0))
        self.assertTrue(np.any(tensor[:, 219:] == 1.0))

    def test_golden_contract_parses_the_authenticated_snapshot_without_reopen(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden, _bundle, _tensor, _logits, path, trusted = self.make_contract(
                Path(temp_dir)
            )
            raw_bytes = path.read_bytes()
            trusted = dataclasses.replace(
                trusted,
                authenticated_files={"golden_contract.json": raw_bytes},
            )
            path.write_bytes(b'{"schema_version":0}')

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("golden contract reopened"),
            ):
                contract = golden.load_golden_contract(
                    path,
                    trusted_release=trusted,
                )

        self.assertEqual(contract.source_sha256, hashlib.sha256(raw_bytes).hexdigest())

    def test_golden_contract_rejects_unknown_nonfinite_wrong_shape_and_identity(self):
        mutations = {
            "unknown field": lambda payload: payload.__setitem__("surprise", True),
            "non-finite": lambda payload: payload.__setitem__("schema_version", float("nan")),
            "tensor_shape": lambda payload: payload.__setitem__("tensor_shape", [63, 438]),
            "component_id": lambda payload: payload.__setitem__("component_id", "wrong"),
            "model_sha256": lambda payload: payload.__setitem__("model_sha256", "f" * 64),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden, _bundle, _tensor, _logits, path, trusted = self.make_contract(root)
            original = json.loads(path.read_text(encoding="utf-8"))
            for expected, mutate in mutations.items():
                with self.subTest(expected=expected):
                    payload = json.loads(json.dumps(original))
                    mutate(payload)
                    path.write_text(
                        json.dumps(payload, sort_keys=True, allow_nan=True),
                        encoding="utf-8",
                    )
                    changed_trust = trusted_release(root, golden_hash=sha256(path))
                    with self.assertRaisesRegex(IntegrityError, expected):
                        golden.load_golden_contract(path, trusted_release=changed_trust)

    def test_golden_verification_rejects_tensor_and_logit_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden, _bundle, tensor, logits, path, trusted = self.make_contract(Path(temp_dir))
            contract = golden.load_golden_contract(path, trusted_release=trusted)
            changed_tensor = tensor.copy()
            changed_tensor[0, 0] += np.float32(1.0)
            with self.assertRaisesRegex(IntegrityError, "tensor SHA-256"):
                golden.verify_golden_result(contract, changed_tensor, logits)
            changed_logits = logits.detach().clone()
            changed_logits[0, 0] += 1.0
            with self.assertRaisesRegex(IntegrityError, "logits SHA-256"):
                golden.verify_golden_result(contract, tensor, changed_logits)

    def test_golden_rejects_non_float32_inputs_before_canonical_hashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden, _bundle, tensor, logits, path, trusted = self.make_contract(
                Path(temp_dir)
            )
            contract = golden.load_golden_contract(path, trusted_release=trusted)
            for dtype in (np.float16, np.float64, np.int32):
                with self.subTest(target="tensor", dtype=str(dtype)):
                    with self.assertRaisesRegex(IntegrityError, "dtype.*float32"):
                        golden.verify_golden_result(
                            contract,
                            tensor.astype(dtype),
                            logits,
                        )
            for dtype in (torch.float16, torch.float64, torch.int64):
                with self.subTest(target="logits", dtype=str(dtype)):
                    with self.assertRaisesRegex(IntegrityError, "dtype.*float32"):
                        golden.verify_golden_result(
                            contract,
                            tensor,
                            logits.to(dtype=dtype),
                        )

            explicit_big_endian = np.asarray(tensor, dtype=">f4")
            golden.verify_golden_result(contract, explicit_big_endian, logits)


if __name__ == "__main__":
    unittest.main()
