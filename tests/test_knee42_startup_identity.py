from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import recognition.realtime.knee42_ivcam as runtime
from recognition.realtime.auto_trigger import AutoTriggerConfig
from recognition.realtime.knee42_integrity import IntegrityError, VerifiedRelease


TRUSTED_TRIGGER_CONFIG_BYTES = b"startup-authenticated-trigger-config"
TRUSTED_PATHS = {
    "requirements-windows-runtime.lock.txt": "1" * 64,
    "golden_contract.json": "2" * 64,
    "auto_trigger_knee_ivcam_local.json": hashlib.sha256(
        TRUSTED_TRIGGER_CONFIG_BYTES
    ).hexdigest(),
    "auto_trigger_provenance.json": "4" * 64,
    "recognition/realtime/auto_trigger.py": "5" * 64,
    "recognition/realtime/knee42_controllers.py": "6" * 64,
    "model/component_manifest.json": "7" * 64,
    "model/integrity_manifest.sha256": "8" * 64,
    "model/best_model.pt": "9" * 64,
    "model/label_map_knee42.json": "0" * 64,
    "model/runtime_config.json": "a" * 64,
    "model/selection_ledger.json": "b" * 64,
    "model/hand_landmarker.task": "c" * 64,
    "model/pose_landmarker.task": "d" * 64,
}


def trusted_release(root: Path) -> VerifiedRelease:
    return VerifiedRelease(
        root=root.resolve(),
        release_version="v1.0.1-v13.1",
        app_version="v13.1",
        component_id="knee42-v12-startup-fixture",
        model_version="v12",
        model_component_manifest_sha256=TRUSTED_PATHS["model/component_manifest.json"],
        label_count=42,
        input_shape=(1, 64, 438),
        source_commit="e" * 40,
        dependency_lock_sha256=TRUSTED_PATHS[
            "requirements-windows-runtime.lock.txt"
        ],
        root_manifest_sha256="f" * 64,
        file_hashes=TRUSTED_PATHS,
        authenticated_files={
            "auto_trigger_knee_ivcam_local.json": TRUSTED_TRIGGER_CONFIG_BYTES,
        },
    )


class FakeBundle:
    component_id = "knee42-v12-startup-fixture"
    model_display_version = "v12"
    labels = [f"K42_{number:02d}" for number in range(1, 43)]
    display_text = {label: label for label in labels}
    mean = np.zeros(219, dtype=np.float32)
    std = np.ones(219, dtype=np.float32)

    def forward_prepared(self, prepared):
        if tuple(prepared.shape) != (64, 438):
            raise AssertionError(prepared.shape)
        return torch.arange(42, dtype=torch.float32).reshape(1, 42)


class FakeDetectorContext:
    def __init__(self, order):
        self.order = order

    def __enter__(self):
        self.order.append("detector")
        return self

    def __exit__(self, *_args):
        return False


class StartupRecordTests(unittest.TestCase):
    def require_api(self):
        builder = getattr(runtime, "build_startup_record", None)
        serializer = getattr(runtime, "serialize_startup_record", None)
        self.assertTrue(callable(builder), "build_startup_record is missing")
        self.assertTrue(callable(serializer), "serialize_startup_record is missing")
        return builder, serializer

    def test_startup_record_has_exact_schema_all_hashes_and_full_trigger_config(self):
        builder, serializer = self.require_api()
        release = trusted_release(Path("release"))
        config = AutoTriggerConfig(
            start_motion_threshold=0.015,
            blank_motion_threshold=0.022,
        )
        record = builder(
            release,
            config,
            trusted_labels=runtime.LABELS,
            clock_policy="video_source_timestamps",
            current_clock_mode="pending_video_source_timestamp",
            rotation_requested="auto",
            rotation_resolved=90,
            input_mirror=False,
            display_mirror=True,
        )
        line = serializer(record)
        parsed = json.loads(line)

        self.assertNotIn("\n", line)
        self.assertEqual(
            line,
            json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ),
        )
        self.assertEqual(
            set(parsed),
            {
                "schema_version",
                "event",
                "entry_module",
                "release_version",
                "app_version",
                "component_id",
                "model_version",
                "source_commit",
                "label_count",
                "labels",
                "input_shape",
                "hashes",
                "clock",
                "orientation",
                "trigger_config",
            },
        )
        self.assertEqual(parsed["event"], "knee42_startup")
        self.assertEqual(parsed["entry_module"], "recognition.realtime.knee42_ivcam")
        self.assertEqual(parsed["component_id"], release.component_id)
        self.assertEqual(parsed["model_version"], release.model_version)
        self.assertEqual(parsed["label_count"], 42)
        self.assertEqual(parsed["labels"], runtime.LABELS)
        self.assertEqual(parsed["input_shape"], [1, 64, 438])
        self.assertEqual(
            set(parsed["hashes"]),
            {
                "root_manifest_sha256",
                "component_manifest_sha256",
                "component_integrity_manifest_sha256",
                "model_sha256",
                "runtime_config_sha256",
                "selection_ledger_sha256",
                "hand_landmarker_task_sha256",
                "pose_landmarker_task_sha256",
                "golden_contract_sha256",
                "trigger_config_sha256",
                "trigger_provenance_sha256",
                "trigger_source_sha256",
                "trigger_controller_sha256",
                "dependency_lock_sha256",
            },
        )
        self.assertEqual(
            parsed["hashes"]["component_manifest_sha256"],
            release.model_component_manifest_sha256,
        )
        self.assertEqual(parsed["hashes"]["model_sha256"], "9" * 64)
        self.assertEqual(
            parsed["clock"],
            {
                "policy": "video_source_timestamps",
                "current_mode": "pending_video_source_timestamp",
            },
        )
        self.assertEqual(
            parsed["orientation"],
            {
                "rotation_requested": "auto",
                "rotation_resolved": 90,
                "input_mirror": False,
                "display_mirror": True,
            },
        )
        self.assertEqual(parsed["trigger_config"], asdict(config))
        self.assertEqual(len(parsed["trigger_config"]), 25)

    def test_startup_serializer_rejects_nonfinite_values(self):
        _builder, serializer = self.require_api()
        with self.assertRaisesRegex(ValueError, "JSON compliant|Out of range"):
            serializer({"event": "knee42_startup", "bad": float("nan")})

    def test_startup_trigger_config_requires_exact_auto_trigger_type(self):
        builder, _serializer = self.require_api()
        lookalike_type = dataclasses.make_dataclass(
            "LookalikeTriggerConfig",
            [
                (
                    f"field_{index}",
                    float,
                    dataclasses.field(default=0.0),
                )
                for index in range(25)
            ],
            frozen=True,
        )

        with self.assertRaisesRegex(IntegrityError, "AutoTriggerConfig"):
            builder(
                trusted_release(Path("release")),
                lookalike_type(),
                trusted_labels=runtime.LABELS,
                clock_policy="video_source_timestamps",
                current_clock_mode="video_source_timestamp",
                rotation_requested=0,
                rotation_resolved=0,
                input_mirror=False,
                display_mirror=False,
            )


class StartupOrderingTests(unittest.TestCase):
    def test_self_test_api_rejects_non_cpu_before_release_access(self):
        with mock.patch.object(runtime, "verify_release_root") as root_verify:
            with self.assertRaisesRegex(IntegrityError, "CPU"):
                runtime.run_self_test(
                    Path("release/model"),
                    device=torch.device("cuda"),
                    expected_root_manifest_sha256="f" * 64,
                )

        root_verify.assert_not_called()

    def test_self_test_emits_after_trust_validation_before_model_and_detector(self):
        order = []
        lines = []
        config = AutoTriggerConfig(
            start_motion_threshold=0.015,
            blank_motion_threshold=0.022,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "model"
            bundle_dir.mkdir()
            release = trusted_release(root)

            def root_verify(*_args, **_kwargs):
                order.append("root")
                return release

            def provenance(*_args, **_kwargs):
                order.append("provenance")
                return {}

            def config_loader(*_args, **_kwargs):
                order.append("config")
                return config

            def golden_loader(*_args, **_kwargs):
                order.append("golden_structure")
                return SimpleNamespace()

            def labels_loader(*_args, **_kwargs):
                order.append("labels")
                return list(runtime.LABELS)

            def sink(line):
                order.append("startup")
                lines.append(line)

            def model_loader(*_args, **_kwargs):
                order.append("model")
                return FakeBundle()

            def golden_verifier(*_args, **_kwargs):
                order.append("golden_execution")

            with (
                mock.patch.object(runtime, "verify_release_root", side_effect=root_verify),
                mock.patch.object(
                    runtime, "verify_auto_trigger_provenance", side_effect=provenance
                ),
                mock.patch.object(
                    runtime, "load_formal_auto_trigger_config", side_effect=config_loader
                ),
                mock.patch.object(
                    runtime, "load_golden_contract", side_effect=golden_loader, create=True
                ),
                mock.patch.object(
                    runtime, "load_verified_labels", side_effect=labels_loader, create=True
                ),
                mock.patch.object(runtime, "load_bundle", side_effect=model_loader),
                mock.patch.object(
                    runtime, "verify_golden_result", side_effect=golden_verifier, create=True
                ),
            ):
                result = runtime.run_self_test(
                    bundle_dir,
                    device=torch.device("cpu"),
                    expected_root_manifest_sha256="f" * 64,
                    detector_factory=lambda *_args, **_kwargs: FakeDetectorContext(order),
                    startup_sink=sink,
                )

        self.assertEqual(
            order,
            [
                "root",
                "provenance",
                "config",
                "labels",
                "golden_structure",
                "startup",
                "model",
                "golden_execution",
                "detector",
            ],
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["component_id"], release.component_id)
        self.assertTrue(result["software_contract_verified"])
        self.assertFalse(result["accuracy_evidence"])

    def test_root_anchor_failure_emits_no_startup_event_and_opens_nothing(self):
        lines = []
        with (
            mock.patch.object(
                runtime,
                "verify_release_root",
                side_effect=IntegrityError("root anchor mismatch"),
            ),
            mock.patch.object(runtime, "load_bundle") as model_loader,
            mock.patch.object(runtime, "open_video") as source_loader,
        ):
            with self.assertRaisesRegex(IntegrityError, "root anchor mismatch"):
                runtime.run_self_test(
                    Path("release/model"),
                    device=torch.device("cpu"),
                    expected_root_manifest_sha256="f" * 64,
                    startup_sink=lines.append,
                )
        self.assertEqual(lines, [])
        model_loader.assert_not_called()
        source_loader.assert_not_called()

    def test_video_identity_emits_before_model_deserialize_with_source_clock(self):
        order = []
        lines = []
        config = AutoTriggerConfig(
            start_motion_threshold=0.015,
            blank_motion_threshold=0.022,
        )

        class Source:
            status = "video:fixture.mp4"
            fps = 30.0
            clock_mode = "video_source_timestamp"
            input_mirror = False
            resolved_rotation = 270
            released = False

            def release(self):
                self.released = True

        source = Source()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "model"
            bundle_dir.mkdir()
            release = trusted_release(root)
            with (
                mock.patch.object(runtime, "verify_release_root", return_value=release),
                mock.patch.object(runtime, "verify_auto_trigger_provenance", return_value={}),
                mock.patch.object(
                    runtime, "load_formal_auto_trigger_config", return_value=config
                ),
                mock.patch.object(
                    runtime, "load_golden_contract", return_value=SimpleNamespace(), create=True
                ),
                mock.patch.object(
                    runtime, "load_verified_labels", return_value=list(runtime.LABELS), create=True
                ),
                mock.patch.object(runtime, "open_video", return_value=source),
                mock.patch.object(
                    runtime,
                    "load_bundle",
                    side_effect=lambda *_args, **_kwargs: (
                        order.append("model"),
                        (_ for _ in ()).throw(RuntimeError("stop after identity")),
                    )[1],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after identity"):
                    runtime.run_capture(
                        bundle_dir,
                        mode="auto",
                        camera_index=None,
                        video=Path("fixture.mp4"),
                        device=torch.device("cpu"),
                        headless=True,
                        max_frames=1,
                        rotation="auto",
                        expected_root_manifest_sha256="f" * 64,
                        startup_sink=lambda line: (order.append("startup"), lines.append(line)),
                    )

        self.assertEqual(order, ["startup", "model"])
        self.assertTrue(source.released)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["model_version"], release.model_version)
        self.assertEqual(parsed["clock"]["current_mode"], "video_source_timestamp")
        self.assertEqual(parsed["orientation"]["rotation_resolved"], 270)

    def test_startup_sink_failure_releases_source_before_model_deserialize(self):
        class Source:
            clock_mode = "video_source_timestamp"
            input_mirror = False
            resolved_rotation = 0

            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        source = Source()
        config = AutoTriggerConfig(
            start_motion_threshold=0.015,
            blank_motion_threshold=0.022,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle_dir = root / "model"
            bundle_dir.mkdir()
            release = trusted_release(root)
            with (
                mock.patch.object(runtime, "verify_release_root", return_value=release),
                mock.patch.object(runtime, "verify_auto_trigger_provenance", return_value={}),
                mock.patch.object(
                    runtime,
                    "load_formal_auto_trigger_config",
                    return_value=config,
                ),
                mock.patch.object(
                    runtime,
                    "load_golden_contract",
                    return_value=SimpleNamespace(),
                ),
                mock.patch.object(
                    runtime,
                    "load_verified_labels",
                    return_value=list(runtime.LABELS),
                ),
                mock.patch.object(runtime, "open_video", return_value=source),
                mock.patch.object(runtime, "load_bundle") as model_loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "startup sink failed"):
                    runtime.run_capture(
                        bundle_dir,
                        mode="auto",
                        camera_index=None,
                        video=Path("fixture.mp4"),
                        device=torch.device("cpu"),
                        headless=True,
                        max_frames=1,
                        expected_root_manifest_sha256="f" * 64,
                        startup_sink=lambda _line: (_ for _ in ()).throw(
                            RuntimeError("startup sink failed")
                        ),
                    )

        self.assertTrue(source.released)
        model_loader.assert_not_called()


class StaticVersionTests(unittest.TestCase):
    def test_version_needs_no_bundle_and_claims_no_component_or_model_identity(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as stopped:
                runtime.main(["--version"])

        self.assertEqual(stopped.exception.code, 0)
        text = stdout.getvalue().strip()
        self.assertIn("Knee42 app v13.1", text)
        self.assertIn("release v1.0.1-v13.1", text)
        self.assertNotIn("component", text.lower())
        self.assertNotIn("model", text.lower())
        self.assertEqual(stderr.getvalue(), "")

    def test_formal_cli_exposes_no_component_or_model_identity_override(self):
        option_strings = {
            option
            for action in runtime.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {
                "--model-version",
                "--component-id",
                "--label-count",
                "--input-shape",
            }.isdisjoint(option_strings)
        )

    def test_cli_self_test_auto_forces_cpu_even_when_cuda_is_available(self):
        captured_devices = []

        def self_test(*_args, **kwargs):
            captured_devices.append(kwargs["device"])
            return {"software_contract_verified": True}

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(runtime, "run_self_test", side_effect=self_test),
            redirect_stdout(io.StringIO()),
        ):
            runtime.main(
                [
                    "--bundle",
                    "release/model",
                    "--root-manifest-sha256",
                    "f" * 64,
                    "--self-test",
                    "--device",
                    "auto",
                ]
            )

        self.assertEqual(captured_devices, [torch.device("cpu")])


class VerifiedLabelMapTests(unittest.TestCase):
    def require_loader(self):
        loader = getattr(runtime, "load_verified_labels", None)
        self.assertTrue(callable(loader), "load_verified_labels is missing")
        return loader

    def test_verified_label_map_returns_exact_ordered_42_labels(self):
        loader = self.require_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            path = model / "label_map_knee42.json"
            payload = {
                "idx_to_label": list(runtime.LABELS),
                "label_to_idx": {
                    label: index for index, label in enumerate(runtime.LABELS)
                },
            }
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            release = trusted_release(root)
            hashes = dict(release.file_hashes)
            hashes["model/label_map_knee42.json"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            release = dataclasses.replace(release, file_hashes=hashes)
            release = dataclasses.replace(
                release,
                authenticated_files={
                    "model/label_map_knee42.json": path.read_bytes(),
                },
            )

            labels = loader(root, trusted_release=release)

        self.assertEqual(labels, runtime.LABELS)

    def test_verified_label_map_parses_the_authenticated_snapshot_without_reopen(self):
        loader = self.require_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            path = model / "label_map_knee42.json"
            raw_bytes = json.dumps(
                {
                    "idx_to_label": list(runtime.LABELS),
                    "label_to_idx": {
                        label: index for index, label in enumerate(runtime.LABELS)
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
            path.write_bytes(raw_bytes)
            release = trusted_release(root)
            hashes = dict(release.file_hashes)
            hashes["model/label_map_knee42.json"] = hashlib.sha256(
                raw_bytes
            ).hexdigest()
            release = dataclasses.replace(
                release,
                file_hashes=hashes,
                authenticated_files={"model/label_map_knee42.json": raw_bytes},
            )
            path.write_bytes(b'{"idx_to_label":[]}')

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("verified label map reopened"),
            ):
                labels = loader(root, trusted_release=release)

        self.assertEqual(labels, runtime.LABELS)

    def test_verified_label_map_rejects_wrong_duplicate_non42_and_inconsistent_maps(self):
        loader = self.require_loader()
        mutations = {
            "wrong": lambda payload: payload["idx_to_label"].__setitem__(0, "K42_99"),
            "duplicate": lambda payload: payload["idx_to_label"].__setitem__(41, "K42_01"),
            "42": lambda payload: payload["idx_to_label"].pop(),
            "inconsistent": lambda payload: payload["label_to_idx"].__setitem__("K42_01", 41),
            "unknown": lambda payload: payload.__setitem__("surprise", True),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            path = model / "label_map_knee42.json"
            for expected, mutate in mutations.items():
                with self.subTest(expected=expected):
                    payload = {
                        "idx_to_label": list(runtime.LABELS),
                        "label_to_idx": {
                            label: index for index, label in enumerate(runtime.LABELS)
                        },
                    }
                    mutate(payload)
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    release = trusted_release(root)
                    hashes = dict(release.file_hashes)
                    hashes["model/label_map_knee42.json"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    release = dataclasses.replace(release, file_hashes=hashes)
                    release = dataclasses.replace(
                        release,
                        authenticated_files={
                            "model/label_map_knee42.json": path.read_bytes(),
                        },
                    )
                    with self.assertRaisesRegex(IntegrityError, expected):
                        loader(root, trusted_release=release)


if __name__ == "__main__":
    unittest.main()
