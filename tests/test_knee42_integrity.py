from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

import recognition.realtime.knee42_ivcam as knee42_ivcam
import recognition.realtime.knee42_integrity as knee42_integrity
from recognition.realtime.knee42_integrity import (
    AssetSpec,
    IntegrityError,
    ReleaseSpec,
    VerifiedRelease,
    load_release_spec,
    parse_sha256_manifest,
    verify_release_root,
)


REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "packaging" / "knee42_ivcam" / "release_spec.json"
VERSION_FIELDS = {
    "release_version": "v1.0.1-v13.1",
    "app_version": "v13.1",
    "model_version": "v11",
    "label_count": 42,
    "input_shape": [1, 64, 438],
    "source_commit": "a" * 40,
    "dependency_lock_sha256": "b" * 64,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def rewrite_root_manifest(root: Path) -> None:
    members = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != "integrity_manifest.sha256"
    )
    (root / "integrity_manifest.sha256").write_text(
        "".join(f"{sha256(release_path(root, member))}  {member}\n" for member in members),
        encoding="ascii",
    )


def make_valid_root(root: Path) -> tuple[Path, ReleaseSpec]:
    canonical = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    canonical_spec = load_release_spec(SPEC_PATH)
    required = set(canonical_spec.required_release_root)
    required.update(f"model/{name}" for name in canonical_spec.required_model_layout)
    for relative in sorted(required):
        path = release_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative not in {
            "VERSION_MANIFEST.json",
            "packaging/knee42_ivcam/release_spec.json",
        }:
            path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))

    for filename in canonical["model_files"]:
        canonical["model_files"][filename] = sha256(root / "model" / filename)
    for asset_name in ("hand_landmarker_task", "pose_landmarker_task"):
        filename = canonical["assets"][asset_name]["filename"]
        canonical["assets"][asset_name]["sha256"] = sha256(root / "model" / filename)
    packaged_spec = release_path(root, "packaging/knee42_ivcam/release_spec.json")
    packaged_spec.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    spec = load_release_spec(packaged_spec)

    version = dict(VERSION_FIELDS)
    version["dependency_lock_sha256"] = sha256(
        root / "requirements-windows-runtime.lock.txt"
    )
    (root / "VERSION_MANIFEST.json").write_text(
        json.dumps(version, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    rewrite_root_manifest(root)
    return root, spec


class ReleaseSpecTests(unittest.TestCase):
    def test_release_spec_pins_fixed_versions_artifacts_assets_and_licenses(self):
        spec = load_release_spec(SPEC_PATH)

        normalized_spec = SPEC_PATH.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(
            hashlib.sha256(normalized_spec).hexdigest(),
            knee42_integrity.CANONICAL_RELEASE_SPEC_SHA256,
        )
        self.assertIsInstance(spec, ReleaseSpec)
        self.assertEqual(spec.release_version, "v1.0.1-v13.1")
        self.assertEqual(spec.app_version, "v13.1")
        self.assertEqual(spec.model_version, "v11")
        self.assertEqual(spec.label_count, 42)
        self.assertEqual(spec.input_shape, (1, 64, 438))
        self.assertEqual(
            dict(spec.artifact_names),
            {
                "source_runtime": "Knee42-v13.1-source-runtime.zip",
                "windows_x64": "Knee42-v13.1-windows-x64.zip",
            },
        )

        hand = spec.assets["hand_landmarker_task"]
        pose = spec.assets["pose_landmarker_task"]
        archive = spec.assets["model_archive"]
        self.assertIsInstance(hand, AssetSpec)
        self.assertEqual(
            (hand.url, hand.sha256, hand.license_id),
            (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task",
                "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
                "Apache-2.0",
            ),
        )
        self.assertEqual(
            (pose.url, pose.sha256, pose.license_id),
            (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
                "Apache-2.0",
            ),
        )
        self.assertEqual(
            (archive.url, archive.sha256, archive.license_id),
            (
                "https://github.com/Mikullee/Sign-Language-Recognition-Gemma/releases/"
                "download/v1.0.0-v13/knee42-model-v11.zip",
                "af45a4a50fc67755dd86be1b47fe975120e47a1b9f6850232e294685dd4ac8df",
                "CC-BY-NC-4.0",
            ),
        )
        self.assertEqual(spec.license_identifiers["application"], "MIT")
        self.assertEqual(spec.license_identifiers["knee42_model"], "CC-BY-NC-4.0")
        self.assertEqual(spec.license_identifiers["mediapipe_tasks"], "Apache-2.0")

    def test_release_spec_pins_corrected_inner_model_hashes(self):
        spec = load_release_spec(SPEC_PATH)

        self.assertEqual(
            dict(spec.model_files),
            {
                "best_model.pt": "8e35adedae1a03ad5644872769821d2966bb7613b5a1e070996929c7e5f2e492",
                "display_text_map.json": "a2d2e008cf6232b29ee04596e1e1bb418ccf0b0587f41e42471cf22e3b2073a3",
                "feature_config.json": "c9670b77a6ab44d766559497e6dbf61a8da9ebfeca45e4b77c0640e596d0a5dc",
                "label_map_knee42.json": "18c8121f8cdfafaf957ba07c7b3181d51055ffdd71493ba27b91c2c7260339b9",
                "standardizer_train_only.npz": "c252b4fdc9fa83179a75bb4726bd0062ebbd796a3beae35f3d47483d2456c391",
            },
        )

    def test_release_spec_pins_required_release_and_model_layouts(self):
        spec = load_release_spec(SPEC_PATH)

        self.assertIn("VERSION_MANIFEST.json", spec.required_release_root)
        self.assertIn(
            "packaging/knee42_ivcam/release_spec.json",
            spec.required_release_root,
        )
        self.assertEqual(
            set(spec.required_model_layout),
            {
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
            },
        )

    def test_release_spec_types_are_deeply_immutable(self):
        spec = load_release_spec(SPEC_PATH)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.label_count = 41  # type: ignore[misc]
        with self.assertRaises(TypeError):
            spec.assets["extra"] = spec.assets["model_archive"]  # type: ignore[index]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.assets["model_archive"].sha256 = "0" * 64  # type: ignore[misc]

    def test_release_spec_rejects_malformed_schema_types_and_values(self):
        canonical = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        mutations = {
            "unknown field": lambda payload: payload.__setitem__("surprise", True),
            "missing field": lambda payload: payload.pop("release_version"),
            "schema_version": lambda payload: payload.__setitem__("schema_version", 2),
            "label_count": lambda payload: payload.__setitem__("label_count", True),
            "input_shape": lambda payload: payload.__setitem__("input_shape", [1, 64, "438"]),
            "asset SHA-256": lambda payload: payload["assets"]["hand_landmarker_task"].__setitem__(
                "sha256", "bad"
            ),
            "asset URL": lambda payload: payload["assets"]["pose_landmarker_task"].__setitem__(
                "url", "http://example.invalid/task"
            ),
            "unsafe layout": lambda payload: payload["required_layouts"][
                "release_root"
            ].append("../escape"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for expected, mutate in mutations.items():
                with self.subTest(expected=expected):
                    payload = json.loads(json.dumps(canonical))
                    mutate(payload)
                    path = Path(temp_dir) / f"{len(expected)}-{expected.replace(' ', '-')}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(IntegrityError, re.escape(expected)):
                        load_release_spec(path)


class ManifestParserTests(unittest.TestCase):
    def parse(self, text: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "integrity_manifest.sha256"
            path.write_text(text, encoding="ascii")
            return parse_sha256_manifest(path)

    def test_parser_accepts_sha256sum_text_and_normalizes_separators(self):
        parsed = self.parse(f"{'a' * 64}  folder\\file.bin\n{'b' * 64} *other.bin\n")

        self.assertEqual(
            dict(parsed),
            {"folder/file.bin": "a" * 64, "other.bin": "b" * 64},
        )

    def test_parser_rejects_malformed_lines_and_hashes(self):
        malformed = (
            "not-a-manifest-entry\n",
            f"{'a' * 63}  file.bin\n",
            f"{'g' * 64}  file.bin\n",
            f"{'a' * 64}  \n",
        )
        for text in malformed:
            with self.subTest(text=text):
                with self.assertRaisesRegex(IntegrityError, "line 1"):
                    self.parse(text)

    def test_parser_rejects_absolute_and_traversal_paths(self):
        unsafe_paths = (
            "/absolute.bin",
            "C:\\absolute.bin",
            "\\\\server\\share\\absolute.bin",
            "../escape.bin",
            "folder/../../escape.bin",
            "folder\\..\\escape.bin",
        )
        for unsafe in unsafe_paths:
            with self.subTest(path=unsafe):
                with self.assertRaisesRegex(IntegrityError, re.escape(unsafe)):
                    self.parse(f"{'a' * 64}  {unsafe}\n")

    def test_parser_rejects_duplicate_normalized_paths(self):
        with self.assertRaisesRegex(IntegrityError, "duplicate.*folder/file.bin"):
            self.parse(
                f"{'a' * 64}  folder/file.bin\n"
                f"{'b' * 64}  folder\\file.bin\n"
            )


class ReleaseRootVerificationTests(unittest.TestCase):
    def test_missing_manifest_names_the_missing_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(IntegrityError, "integrity_manifest.sha256"):
                verify_release_root(Path(temp_dir))

    def test_root_manifest_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = make_valid_root(Path(temp_dir))
            missing = root / "model" / "best_model.pt"
            missing.unlink()

            with self.assertRaisesRegex(IntegrityError, "missing.*model/best_model.pt"):
                verify_release_root(root, spec=spec)

    def test_root_manifest_rejects_unlisted_surprise_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = make_valid_root(Path(temp_dir))
            (root / "surprise.exe").write_bytes(b"x")

            with self.assertRaisesRegex(IntegrityError, "unexpected.*surprise.exe"):
                verify_release_root(root, spec=spec)

    def test_root_manifest_rejects_changed_byte_and_names_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = make_valid_root(Path(temp_dir))
            changed = root / "MODEL_CARD.md"
            changed.write_bytes(changed.read_bytes() + b"x")

            with self.assertRaisesRegex(IntegrityError, "mismatch.*MODEL_CARD.md"):
                verify_release_root(root, spec=spec)

    def test_regenerated_manifest_cannot_bless_changed_pinned_model_or_task(self):
        pinned_paths = ("model/best_model.pt", "model/hand_landmarker.task")
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            for index, relative in enumerate(pinned_paths):
                with self.subTest(path=relative):
                    root, spec = make_valid_root(parent / str(index))
                    path = release_path(root, relative)
                    path.write_bytes(path.read_bytes() + b"changed")
                    rewrite_root_manifest(root)

                    with self.assertRaisesRegex(
                        IntegrityError,
                        rf"canonical.*{re.escape(relative)}",
                    ):
                        verify_release_root(root, spec=spec)

    def test_regenerated_manifest_cannot_bless_changed_packaged_release_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = make_valid_root(Path(temp_dir))
            packaged_spec = release_path(
                root, "packaging/knee42_ivcam/release_spec.json"
            )
            packaged_spec.write_bytes(packaged_spec.read_bytes() + b"\n")
            rewrite_root_manifest(root)

            with self.assertRaisesRegex(
                IntegrityError,
                "canonical.*packaging/knee42_ivcam/release_spec.json",
            ):
                verify_release_root(root, spec=spec)

    def test_default_verifier_uses_the_canonical_tracked_spec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, _fixture_spec = make_valid_root(Path(temp_dir))

            with self.assertRaisesRegex(
                IntegrityError,
                "canonical.*packaging/knee42_ivcam/release_spec.json",
            ):
                verify_release_root(root)

    def test_packaged_default_spec_cannot_authorize_its_own_changed_pins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, _fixture_spec = make_valid_root(Path(temp_dir))
            packaged_spec = release_path(
                root, "packaging/knee42_ivcam/release_spec.json"
            )

            with mock.patch.object(
                knee42_integrity,
                "DEFAULT_RELEASE_SPEC_PATH",
                packaged_spec,
            ):
                with self.assertRaisesRegex(
                    IntegrityError,
                    "canonical release spec SHA-256",
                ):
                    verify_release_root(root)

    def test_version_manifest_rejects_wrong_contract_fields(self):
        wrong_values = {
            "release_version": "v1.0.0-v13",
            "app_version": "v13",
            "model_version": "v10",
            "input_shape": [1, 63, 438],
            "label_count": 41,
            "source_commit": "not-a-commit",
            "dependency_lock_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            for index, (field, wrong) in enumerate(wrong_values.items()):
                with self.subTest(field=field):
                    root, spec = make_valid_root(parent / str(index))
                    version_path = root / "VERSION_MANIFEST.json"
                    payload = json.loads(version_path.read_text(encoding="utf-8"))
                    payload[field] = wrong
                    version_path.write_text(json.dumps(payload), encoding="utf-8")
                    rewrite_root_manifest(root)

                    with self.assertRaisesRegex(IntegrityError, re.escape(field)):
                        verify_release_root(root, spec=spec)

    def test_valid_root_returns_immutable_structured_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root, spec = make_valid_root(Path(temp_dir))

            verified = verify_release_root(root, spec=spec)

            self.assertIsInstance(verified, VerifiedRelease)
            self.assertEqual(verified.root, root.resolve())
            self.assertEqual(verified.release_version, spec.release_version)
            self.assertEqual(verified.app_version, spec.app_version)
            self.assertEqual(verified.model_version, spec.model_version)
            self.assertEqual(verified.label_count, 42)
            self.assertEqual(verified.input_shape, (1, 64, 438))
            self.assertEqual(verified.source_commit, "a" * 40)
            self.assertEqual(
                verified.dependency_lock_sha256,
                sha256(root / "requirements-windows-runtime.lock.txt"),
            )
            self.assertEqual(
                verified.root_manifest_sha256,
                sha256(root / "integrity_manifest.sha256"),
            )
            self.assertIn("model/best_model.pt", verified.file_hashes)
            with self.assertRaises(TypeError):
                verified.file_hashes["extra"] = "0" * 64  # type: ignore[index]

    def test_legacy_runtime_reexports_the_shared_integrity_error_and_hash_helper(self):
        from recognition.realtime import knee42_integrity

        self.assertIs(knee42_ivcam.IntegrityError, knee42_integrity.IntegrityError)
        self.assertIs(knee42_ivcam.sha256_file, knee42_integrity.sha256_file)


if __name__ == "__main__":
    unittest.main()
