from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from pathlib import PurePosixPath
from unittest import mock

import recognition.realtime.knee42_integrity as integrity
from recognition.realtime.knee42_integrity import IntegrityError, load_release_spec


REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "packaging" / "knee42_ivcam" / "release_spec.json"
PAYLOAD_NAMES = {
    "best_model.pt",
    "runtime_config.json",
    "feature_config.json",
    "standardizer_train_only.npz",
    "label_map_knee42.json",
    "display_text_map.json",
    "selection_ledger.json",
    "hand_landmarker.task",
    "pose_landmarker.task",
}


def write_component(path: Path, **changes) -> tuple[dict, str]:
    payload_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in sorted(PAYLOAD_NAMES)
    }
    payload = {
        "schema_version": 1,
        "component_id": "knee42-v12-test",
        "model_version": "v12",
        "label_count": 42,
        "input_shape": [1, 64, 438],
        "runtime_config_sha256": payload_hashes["runtime_config.json"],
        "selection_ledger_sha256": payload_hashes["selection_ledger.json"],
        "payload_sha256": payload_hashes,
    }
    payload.update(changes)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def release_path(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(relative).parts)


def rewrite_root_manifest(root: Path) -> str:
    members = sorted(
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix() != "integrity_manifest.sha256"
    )
    manifest = root / "integrity_manifest.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(release_path(root, name).read_bytes()).hexdigest()}  {name}\n"
            for name in members
        ),
        encoding="ascii",
    )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def make_component_release(root: Path, *, model_version: str = "v12"):
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
            "model/component_manifest.json",
            "model/integrity_manifest.sha256",
        }:
            path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))

    for filename in canonical["model_files"]:
        canonical["model_files"][filename] = hashlib.sha256(
            (root / "model" / filename).read_bytes()
        ).hexdigest()
    for asset_name in ("hand_landmarker_task", "pose_landmarker_task"):
        filename = canonical["assets"][asset_name]["filename"]
        canonical["assets"][asset_name]["sha256"] = hashlib.sha256(
            (root / "model" / filename).read_bytes()
        ).hexdigest()
    packaged_spec = release_path(root, "packaging/knee42_ivcam/release_spec.json")
    packaged_spec.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    spec = load_release_spec(packaged_spec)

    payload_hashes = {
        name: hashlib.sha256((root / "model" / name).read_bytes()).hexdigest()
        for name in sorted(PAYLOAD_NAMES)
    }
    (root / "model" / "integrity_manifest.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in payload_hashes.items()),
        encoding="ascii",
    )
    component = {
        "schema_version": 1,
        "component_id": f"knee42-{model_version[1:]}-fixture",
        "model_version": model_version,
        "label_count": 42,
        "input_shape": [1, 64, 438],
        "runtime_config_sha256": payload_hashes["runtime_config.json"],
        "selection_ledger_sha256": payload_hashes["selection_ledger.json"],
        "payload_sha256": payload_hashes,
    }
    component_path = root / "model" / "component_manifest.json"
    component_path.write_text(json.dumps(component, sort_keys=True), encoding="utf-8")
    component_hash = hashlib.sha256(component_path.read_bytes()).hexdigest()
    lock_hash = hashlib.sha256(
        (root / "requirements-windows-runtime.lock.txt").read_bytes()
    ).hexdigest()
    (root / "VERSION_MANIFEST.json").write_text(
        json.dumps(
            {
                "release_version": spec.release_version,
                "app_version": spec.app_version,
                "component_id": component["component_id"],
                "model_version": component["model_version"],
                "model_component_manifest_sha256": component_hash,
                "label_count": 42,
                "input_shape": [1, 64, 438],
                "source_commit": "a" * 40,
                "dependency_lock_sha256": lock_hash,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    root_hash = rewrite_root_manifest(root)
    return spec, component, root_hash


class ReleaseComponentSpecTests(unittest.TestCase):
    def test_release_spec_names_default_component_without_claiming_active_model(self):
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        spec = load_release_spec(SPEC_PATH)

        self.assertNotIn("model_version", raw)
        self.assertEqual(raw["default_model_version"], "v11")
        self.assertEqual(raw["component_manifest_name"], "component_manifest.json")
        self.assertEqual(spec.default_model_version, "v11")
        self.assertEqual(spec.component_manifest_name, "component_manifest.json")
        self.assertFalse(hasattr(spec, "model_version"))
        self.assertIn("component_manifest.json", spec.required_model_layout)

    def test_tracked_default_overlays_match_public_v13_selection_contract(self):
        overlay_root = REPO / "packaging" / "knee42_ivcam" / "default_component"
        self.assertFalse((overlay_root.parent / "runtime_config.json").exists())
        self.assertFalse((overlay_root.parent / "selection_ledger.json").exists())
        expected = {
            "runtime_config.json": (
                "f6cbb71c287f03dc75f29662d4a0daff1bb036712c3ad79d97c15807519f0959"
            ),
            "selection_ledger.json": (
                "0e4a4f300e603cd7bb8f80a4f062691710475df1e621f4c885f0a000a818e28f"
            ),
        }
        loaded = {}
        for name, wanted_hash in expected.items():
            with self.subTest(name=name):
                path = overlay_root / name
                normalized = path.read_bytes().replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(normalized).hexdigest(), wanted_hash)
                loaded[name] = json.loads(normalized.decode("utf-8"))

        runtime = loaded["runtime_config.json"]
        ledger = loaded["selection_ledger.json"]
        self.assertEqual(runtime["sequence_length"], 64)
        self.assertEqual(runtime["model_input_dim"], 438)
        self.assertEqual(runtime["frame_step"], 2)
        self.assertEqual(runtime["stream"], "RGB/color")
        self.assertEqual(ledger["selection_metric"], "dev_macro_top1")
        self.assertEqual(ledger["source_commit"], "f10980bae4549fddcbb3fabebb617e555557172e")
        self.assertNotIn("C:\\", json.dumps(loaded))


class ComponentManifestTests(unittest.TestCase):
    def setUp(self):
        self.loader = getattr(integrity, "load_component_manifest", None)

    def require_loader(self):
        self.assertTrue(
            callable(self.loader),
            "load_component_manifest must implement the component trust boundary",
        )
        return self.loader

    def test_component_manifest_accepts_alternate_compatible_model_and_is_immutable(self):
        loader = self.require_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "component_manifest.json"
            payload, digest = write_component(path)
            component = loader(
                path,
                expected_sha256=digest,
                spec=load_release_spec(SPEC_PATH),
            )

        self.assertIsInstance(component, integrity.ComponentManifest)
        self.assertEqual(component.component_id, "knee42-v12-test")
        self.assertEqual(component.model_version, "v12")
        self.assertEqual(component.label_count, 42)
        self.assertEqual(component.input_shape, (1, 64, 438))
        self.assertEqual(
            component.runtime_config_sha256,
            payload["payload_sha256"]["runtime_config.json"],
        )
        self.assertEqual(
            component.selection_ledger_sha256,
            payload["payload_sha256"]["selection_ledger.json"],
        )
        self.assertEqual(dict(component.payload_sha256), payload["payload_sha256"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            component.model_version = "v13"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            component.payload_sha256["extra.bin"] = "0" * 64  # type: ignore[index]

    def test_raw_sha_is_checked_before_component_json_parse(self):
        loader = self.require_loader()
        spec = load_release_spec(SPEC_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "component_manifest.json"
            path.write_text("{not valid json", encoding="utf-8")
            with mock.patch.object(
                integrity,
                "parse_json_object_bytes",
                side_effect=AssertionError("JSON parse happened before trust check"),
            ) as parser:
                with self.assertRaisesRegex(IntegrityError, "component manifest SHA-256"):
                    loader(
                        path,
                        expected_sha256="f" * 64,
                        spec=spec,
                    )
            parser.assert_not_called()

    def test_component_manifest_rejects_unknown_and_missing_fields(self):
        loader = self.require_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, changes in enumerate((
                {"surprise": True},
                {"component_id": None},
            )):
                with self.subTest(changes=changes):
                    path = root / f"component-{index}.json"
                    payload, _digest = write_component(path)
                    if changes["component_id"] is None if "component_id" in changes else False:
                        payload.pop("component_id")
                    else:
                        payload.update(changes)
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(IntegrityError, "field"):
                        loader(path, expected_sha256=digest, spec=load_release_spec(SPEC_PATH))

    def test_component_manifest_rejects_missing_extra_and_unsafe_payload_names(self):
        loader = self.require_loader()
        mutations = {
            "missing": lambda items: items.pop("best_model.pt"),
            "unexpected": lambda items: items.__setitem__("extra.bin", "a" * 64),
            "unsafe": lambda items: items.__setitem__("../escape", "a" * 64),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (expected, mutate) in enumerate(mutations.items()):
                with self.subTest(expected=expected):
                    path = root / f"payload-{index}.json"
                    payload, _digest = write_component(path)
                    mutate(payload["payload_sha256"])
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(IntegrityError, expected):
                        loader(path, expected_sha256=digest, spec=load_release_spec(SPEC_PATH))

    def test_dedicated_runtime_and_ledger_hashes_must_equal_payload_map(self):
        loader = self.require_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, field in enumerate((
                "runtime_config_sha256",
                "selection_ledger_sha256",
            )):
                with self.subTest(field=field):
                    path = root / f"binding-{index}.json"
                    payload, _digest = write_component(path)
                    payload[field] = "f" * 64
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(IntegrityError, field):
                        loader(path, expected_sha256=digest, spec=load_release_spec(SPEC_PATH))


class VerifiedComponentReleaseTests(unittest.TestCase):
    def test_release_root_returns_identity_only_from_verified_alternate_component(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, component, root_hash = make_component_release(root, model_version="v12")

            verified = integrity.verify_release_root(
                root,
                expected_root_manifest_sha256=root_hash,
                spec=spec,
            )

        self.assertEqual(verified.component_id, component["component_id"])
        self.assertEqual(verified.model_version, "v12")
        self.assertEqual(
            verified.model_component_manifest_sha256,
            hashlib.sha256(
                json.dumps(component, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(verified.label_count, 42)
        self.assertEqual(verified.input_shape, (1, 64, 438))

    def test_version_manifest_cannot_override_verified_component_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _component, _root_hash = make_component_release(root)
            version_path = root / "VERSION_MANIFEST.json"
            version = json.loads(version_path.read_text(encoding="utf-8"))
            version["model_version"] = "v99"
            version_path.write_text(json.dumps(version, sort_keys=True), encoding="utf-8")
            root_hash = rewrite_root_manifest(root)

            with self.assertRaisesRegex(IntegrityError, "model_version mismatch"):
                integrity.verify_release_root(
                    root,
                    expected_root_manifest_sha256=root_hash,
                    spec=spec,
                )

    def test_compatible_alternate_may_reuse_default_version_without_default_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, component, _root_hash = make_component_release(root, model_version="v11")
            model_path = root / "model" / "best_model.pt"
            model_path.write_bytes(b"compatible alternate v11 checkpoint")
            replacement_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            component["payload_sha256"]["best_model.pt"] = replacement_hash
            internal_path = root / "model" / "integrity_manifest.sha256"
            internal = dict(integrity.parse_sha256_manifest(internal_path))
            internal["best_model.pt"] = replacement_hash
            internal_path.write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sorted(internal.items())),
                encoding="ascii",
            )
            component_path = root / "model" / "component_manifest.json"
            component_path.write_text(json.dumps(component, sort_keys=True), encoding="utf-8")
            version_path = root / "VERSION_MANIFEST.json"
            version = json.loads(version_path.read_text(encoding="utf-8"))
            version["model_component_manifest_sha256"] = hashlib.sha256(
                component_path.read_bytes()
            ).hexdigest()
            version_path.write_text(json.dumps(version, sort_keys=True), encoding="utf-8")
            root_hash = rewrite_root_manifest(root)

            verified = integrity.verify_release_root(
                root,
                expected_root_manifest_sha256=root_hash,
                spec=spec,
            )

        self.assertEqual(verified.model_version, "v11")
        self.assertEqual(verified.component_id, component["component_id"])

    def test_component_payload_must_match_root_and_internal_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _component, _root_hash = make_component_release(root)
            runtime = root / "model" / "runtime_config.json"
            runtime.write_text("attacker payload", encoding="utf-8")
            root_hash = rewrite_root_manifest(root)

            with self.assertRaisesRegex(IntegrityError, "component payload SHA-256"):
                integrity.verify_release_root(
                    root,
                    expected_root_manifest_sha256=root_hash,
                    spec=spec,
                )

            spec, component, _root_hash = make_component_release(root)
            component["payload_sha256"]["runtime_config.json"] = "f" * 64
            component["runtime_config_sha256"] = "f" * 64
            component_path = root / "model" / "component_manifest.json"
            component_path.write_text(json.dumps(component, sort_keys=True), encoding="utf-8")
            version_path = root / "VERSION_MANIFEST.json"
            version = json.loads(version_path.read_text(encoding="utf-8"))
            version["model_component_manifest_sha256"] = hashlib.sha256(
                component_path.read_bytes()
            ).hexdigest()
            version_path.write_text(json.dumps(version, sort_keys=True), encoding="utf-8")
            root_hash = rewrite_root_manifest(root)

            with self.assertRaisesRegex(IntegrityError, "component payload SHA-256"):
                integrity.verify_release_root(
                    root,
                    expected_root_manifest_sha256=root_hash,
                    spec=spec,
                )

    def test_component_manifest_bytes_are_root_bound_before_json_parse(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _component, trusted_root_hash = make_component_release(root)
            (root / "model" / "component_manifest.json").write_text(
                "{self blessed but invalid", encoding="utf-8"
            )
            rewrite_root_manifest(root)
            with mock.patch.object(
                integrity,
                "load_component_manifest",
                side_effect=AssertionError("component parser reached"),
            ) as component_loader:
                with self.assertRaisesRegex(IntegrityError, "root integrity manifest SHA-256"):
                    integrity.verify_release_root(
                        root,
                        expected_root_manifest_sha256=trusted_root_hash,
                        spec=spec,
                    )
            component_loader.assert_not_called()

    def test_component_manifest_rejects_noncanonical_hash_shape_and_identity(self):
        loader = integrity.load_component_manifest
        mutations = {
            "SHA-256": lambda payload: payload["payload_sha256"].__setitem__(
                "best_model.pt", "A" * 64
            ),
            "component_id": lambda payload: payload.__setitem__("component_id", "../bad"),
            "model_version": lambda payload: payload.__setitem__("model_version", "latest"),
            "label_count": lambda payload: payload.__setitem__("label_count", 41),
            "input_shape": lambda payload: payload.__setitem__("input_shape", [1, 32, 438]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (expected, mutate) in enumerate(mutations.items()):
                with self.subTest(expected=expected):
                    path = root / f"invalid-{index}.json"
                    payload, _digest = write_component(path)
                    mutate(payload)
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(IntegrityError, expected):
                        loader(path, expected_sha256=digest, spec=load_release_spec(SPEC_PATH))


class AuthenticatedSnapshotTests(unittest.TestCase):
    def test_release_spec_parses_and_hashes_one_byte_snapshot(self):
        original = SPEC_PATH.read_bytes()
        swapped = original.replace(b'"app_version": "v13.1"', b'"app_version": "v99.9"')
        self.assertNotEqual(swapped, original)
        calls = 0

        def swap_on_second_read(candidate: Path) -> bytes:
            nonlocal calls
            calls += 1
            return original if calls == 1 else swapped

        with mock.patch.object(
            Path,
            "read_bytes",
            autospec=True,
            side_effect=swap_on_second_read,
        ):
            loaded = integrity.load_release_spec(SPEC_PATH)

        self.assertEqual(loaded.app_version, "v13.1")
        self.assertEqual(
            loaded.source_sha256,
            hashlib.sha256(original.replace(b"\r\n", b"\n")).hexdigest(),
        )
        self.assertEqual(calls, 1)

    def test_public_authenticated_reader_hashes_the_single_read_bytes(self):
        reader = getattr(integrity, "read_authenticated_bytes", None)
        self.assertTrue(
            callable(reader),
            "read_authenticated_bytes must expose the authenticated snapshot boundary",
        )
        if not callable(reader):
            return
        original = b'{"trusted":true}\n'
        swapped = b'{"trusted":false}\n'
        expected = hashlib.sha256(original).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "identity.json"
            path.write_bytes(original)
            calls = 0

            def swap_on_second_read(candidate: Path) -> bytes:
                nonlocal calls
                calls += 1
                return original if calls == 1 else swapped

            with mock.patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=swap_on_second_read,
            ):
                authenticated = reader(
                    path,
                    expected_sha256=expected,
                    description="identity fixture",
                )

        self.assertEqual(authenticated, original)
        self.assertEqual(calls, 1)

    def test_release_verification_reads_every_authenticated_file_once(self):
        reader = getattr(integrity, "read_authenticated_bytes", None)
        self.assertTrue(callable(reader), "authenticated reader is missing")
        if not callable(reader):
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec, _component, root_hash = make_component_release(root)
            original_reader = reader
            seen: dict[Path, int] = {}

            def counted(path, **kwargs):
                resolved = Path(path).resolve()
                seen[resolved] = seen.get(resolved, 0) + 1
                if seen[resolved] > 1:
                    raise AssertionError(f"authenticated file reopened: {resolved}")
                return original_reader(path, **kwargs)

            with mock.patch.object(
                integrity,
                "read_authenticated_bytes",
                side_effect=counted,
            ):
                verified = integrity.verify_release_root(
                    root,
                    expected_root_manifest_sha256=root_hash,
                    spec=spec,
                )

            expected_paths = {
                (root / "integrity_manifest.sha256").resolve(),
                *{
                    release_path(root, relative).resolve()
                    for relative in verified.file_hashes
                },
            }
            self.assertEqual(set(seen), expected_paths)
            self.assertTrue(all(count == 1 for count in seen.values()))
            self.assertEqual(
                verified.authenticated_files["model/component_manifest.json"],
                (root / "model" / "component_manifest.json").read_bytes(),
            )
            with self.assertRaises(TypeError):
                verified.authenticated_files["VERSION_MANIFEST.json"] = b"swapped"


if __name__ == "__main__":
    unittest.main()
