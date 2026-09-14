from __future__ import annotations

import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_mac_auto_trigger_package import (
    ANNOTATED_VIDEOS,
    BROWSER_ASSET_PATHS,
    PYTHON_MODEL_PATHS,
    validate_annotation_links,
    verify_archive,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_contract_names_all_runtime_and_benchmark_assets() -> None:
    assert PYTHON_MODEL_PATHS == (
        Path("models/hand_landmarker.task"),
        Path("models/pose_landmarker.task"),
    )
    assert set(BROWSER_ASSET_PATHS) == {
        Path("webservice/vendor/mediapipe/vision_bundle.mjs"),
        Path("webservice/vendor/mediapipe/wasm/vision_wasm_internal.js"),
        Path("webservice/vendor/mediapipe/wasm/vision_wasm_internal.wasm"),
        Path("webservice/vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js"),
        Path("webservice/vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm"),
        Path("webservice/vendor/mediapipe/hand_landmarker.task"),
        Path("webservice/vendor/mediapipe/pose_landmarker_lite.task"),
    }
    assert ANNOTATED_VIDEOS == ("你好.mp4", "我肚子餓.mp4", "晚安.mp4")


def test_annotation_links_must_resolve_inside_package(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "data" / "annotations"
    video_dir = tmp_path / "data" / "videos" / "auto_trigger"
    annotation_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    csv_path = annotation_dir / "auto_trigger_three_videos.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "expected_label", "start_sec", "end_sec"])
        writer.writerow(["../videos/auto_trigger/你好.mp4", "你好", "1.333", "4.430"])

    with pytest.raises(FileNotFoundError, match="你好.mp4"):
        validate_annotation_links(tmp_path)

    (video_dir / "你好.mp4").write_bytes(b"video")
    validate_annotation_links(tmp_path)


def test_verify_archive_rejects_missing_required_payload(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Knee42-Mac-AutoTrigger/MAC_PACKAGE_MANIFEST.json", "{}")

    with pytest.raises(ValueError, match="missing required package files"):
        verify_archive(archive)


def test_mac_scripts_are_loopback_only_and_use_web_trigger_profile() -> None:
    setup = (ROOT / "scripts" / "setup_mac_auto_trigger.sh").read_text(encoding="utf-8")
    launch = (ROOT / "scripts" / "run_mac_auto_trigger.sh").read_text(encoding="utf-8")

    assert "Darwin" in setup
    assert "arm64" in setup
    assert "requirements-transformer.txt" in setup
    assert "verify_mac_auto_trigger_package.py" in setup
    assert "--host 127.0.0.1" in launch
    assert "--http" in launch
    assert "configs/auto_trigger_knee_web_live.json" in launch
    assert "0.0.0.0" not in launch


def test_builder_can_be_invoked_by_file_path() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_mac_auto_trigger_package.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_handoff_document_states_scope_and_known_limitations() -> None:
    text = (ROOT / "docs" / "mac_auto_trigger_handoff.md").read_text(encoding="utf-8")

    assert "不需要重訓" in text
    assert "data/annotations/auto_trigger_three_videos.csv" in text
    assert "膝蓋" in text
    assert "timeout" in text
    assert "./scripts/setup_mac_auto_trigger.sh" in text
    assert "./scripts/run_mac_auto_trigger.sh" in text


def test_manifest_marks_candidate_as_not_field_accepted(tmp_path: Path) -> None:
    manifest = {
        "release_status": "auto_trigger_tuning_candidate",
        "field_accepted": False,
        "model_retraining_required": False,
    }
    path = tmp_path / "MAC_PACKAGE_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["field_accepted"] is False
    assert loaded["model_retraining_required"] is False
