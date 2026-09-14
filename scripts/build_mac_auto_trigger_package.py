"""Build and verify the self-contained Knee42 Mac auto-trigger handoff ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    from scripts.fetch_mediapipe_models import BROWSER_ASSETS, PYTHON_MODELS, sha256_file
except ModuleNotFoundError:  # Direct invocation: python scripts/build_mac_auto_trigger_package.py
    from fetch_mediapipe_models import BROWSER_ASSETS, PYTHON_MODELS, sha256_file


PYTHON_MODEL_PATHS = (
    Path("models/hand_landmarker.task"),
    Path("models/pose_landmarker.task"),
)
BROWSER_ASSET_PATHS = (
    Path("webservice/vendor/mediapipe/vision_bundle.mjs"),
    Path("webservice/vendor/mediapipe/wasm/vision_wasm_internal.js"),
    Path("webservice/vendor/mediapipe/wasm/vision_wasm_internal.wasm"),
    Path("webservice/vendor/mediapipe/wasm/vision_wasm_nosimd_internal.js"),
    Path("webservice/vendor/mediapipe/wasm/vision_wasm_nosimd_internal.wasm"),
    Path("webservice/vendor/mediapipe/hand_landmarker.task"),
    Path("webservice/vendor/mediapipe/pose_landmarker_lite.task"),
)
ANNOTATED_VIDEOS = ("你好.mp4", "我肚子餓.mp4", "晚安.mp4")
ANNOTATION_PATH = Path("data/annotations/auto_trigger_three_videos.csv")
PACKAGE_MANIFEST = Path("MAC_PACKAGE_MANIFEST.json")
MODEL_BUNDLE_FILES = (
    "best_model.pt",
    "label_map_knee42.json",
    "display_text_map.json",
    "feature_config.json",
    "runtime_config.json",
    "model_card.json",
    "integrity_manifest.sha256",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _required_release_paths() -> set[Path]:
    paths = {
        PACKAGE_MANIFEST,
        Path("requirements-transformer.txt"),
        Path("configs/auto_trigger_knee_web_live.json"),
        Path("scripts/setup_mac_auto_trigger.sh"),
        Path("scripts/run_mac_auto_trigger.sh"),
        Path("scripts/verify_mac_auto_trigger_package.py"),
        Path("docs/mac_auto_trigger_handoff.md"),
        ANNOTATION_PATH,
    }
    paths.update(PYTHON_MODEL_PATHS)
    paths.update(BROWSER_ASSET_PATHS)
    paths.update(Path("artifacts/realtime/best_current") / name for name in MODEL_BUNDLE_FILES)
    paths.update(Path("data/videos/auto_trigger") / name for name in ANNOTATED_VIDEOS)
    return paths


def validate_annotation_links(package_root: Path) -> None:
    csv_path = package_root / ANNOTATION_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"annotation CSV missing: {csv_path}")
    package_root = package_root.resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("annotation CSV contains no video rows")
    for row in rows:
        relative = row.get("video_path", "").strip()
        target = (csv_path.parent / relative).resolve()
        try:
            target.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"annotation path escapes package: {relative}") from exc
        if not target.is_file():
            raise FileNotFoundError(f"annotated video missing: {target.name}")


def _verify_known_asset(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def validate_release_tree(package_root: Path) -> None:
    missing = sorted(str(path) for path in _required_release_paths() if not (package_root / path).is_file())
    if missing:
        raise ValueError("missing required package files: " + ", ".join(missing))

    for name, spec in PYTHON_MODELS.items():
        _verify_known_asset(package_root / "models" / name, spec["sha256"])
    for name, spec in BROWSER_ASSETS.items():
        _verify_known_asset(package_root / "webservice" / "vendor" / "mediapipe" / name, spec["sha256"])
    _verify_known_asset(
        package_root / "webservice/vendor/mediapipe/hand_landmarker.task",
        PYTHON_MODELS["hand_landmarker.task"]["sha256"],
    )
    _verify_known_asset(
        package_root / "webservice/vendor/mediapipe/pose_landmarker_lite.task",
        PYTHON_MODELS["pose_landmarker.task"]["sha256"],
    )

    bundle_root = package_root / "artifacts/realtime/best_current"
    with (bundle_root / "integrity_manifest.sha256").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            expected, filename = line.split(maxsplit=1)
            _verify_known_asset(bundle_root / filename, expected)
    label_map = json.loads((bundle_root / "label_map_knee42.json").read_text(encoding="utf-8"))
    if len(label_map.get("idx_to_label", [])) != 42:
        raise ValueError("shipped model label map is not Knee42")
    validate_annotation_links(package_root)


def _safe_archive_names(names: list[str]) -> dict[str, str]:
    roots: set[str] = set()
    relative_to_member: dict[str, str] = {}
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe ZIP member: {name}")
        if not pure.parts:
            continue
        roots.add(pure.parts[0])
        if len(pure.parts) > 1 and not name.endswith("/"):
            relative_to_member[PurePosixPath(*pure.parts[1:]).as_posix()] = name
    if len(roots) != 1:
        raise ValueError("package ZIP must contain exactly one top-level directory")
    return relative_to_member


def verify_archive(archive: Path) -> dict:
    with zipfile.ZipFile(archive, "r") as bundle:
        members = _safe_archive_names(bundle.namelist())
        required = {path.as_posix() for path in _required_release_paths()}
        missing = sorted(required - set(members))
        if missing:
            raise ValueError("missing required package files: " + ", ".join(missing))
        manifest = json.loads(bundle.read(members[PACKAGE_MANIFEST.as_posix()]))
        if manifest.get("field_accepted") is not False:
            raise ValueError("candidate package must not claim field acceptance")
        if manifest.get("model_retraining_required") is not False:
            raise ValueError("package must state that model retraining is not required")
        for relative, expected in manifest.get("files", {}).items():
            member = members.get(relative)
            if member is None:
                raise ValueError(f"manifest references missing file: {relative}")
            if _sha256_bytes(bundle.read(member)) != expected:
                raise ValueError(f"archive SHA-256 mismatch for {relative}")
        label_map = json.loads(
            bundle.read(members["artifacts/realtime/best_current/label_map_knee42.json"])
        )
        if len(label_map.get("idx_to_label", [])) != 42:
            raise ValueError("archive model label map is not Knee42")
    return manifest


def _git_output(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=source_root, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _copy_tracked_tree(source_root: Path, destination: Path) -> None:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=source_root, check=True, capture_output=True
    ).stdout
    for raw_name in output.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(os.fsdecode(raw_name))
        source = source_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _package_file_hashes(package_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != PACKAGE_MANIFEST.name:
            result[path.relative_to(package_root).as_posix()] = sha256_file(path)
    return result


def build_package(
    source_root: Path,
    video_dir: Path,
    output_dir: Path,
    *,
    models_dir: Path | None = None,
    vendor_dir: Path | None = None,
) -> tuple[Path, Path]:
    source_root = source_root.resolve()
    video_dir = video_dir.resolve()
    models_dir = (models_dir or source_root / "models").resolve()
    vendor_dir = (vendor_dir or source_root / "webservice/vendor/mediapipe").resolve()
    commit = _git_output(source_root, "rev-parse", "HEAD")
    short_commit = commit[:8]
    branch = _git_output(source_root, "branch", "--show-current")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"Knee42-Mac-AutoTrigger-{short_commit}.zip"
    sidecar = archive.with_suffix(archive.suffix + ".sha256")

    with tempfile.TemporaryDirectory(prefix="knee42-mac-package-") as temp_name:
        package_root = Path(temp_name) / "Knee42-Mac-AutoTrigger"
        package_root.mkdir()
        _copy_tracked_tree(source_root, package_root)

        for relative in PYTHON_MODEL_PATHS:
            source = models_dir / relative.name
            target = package_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative in BROWSER_ASSET_PATHS:
            source = vendor_dir / relative.relative_to("webservice/vendor/mediapipe")
            target = package_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        target_video_dir = package_root / "data/videos/auto_trigger"
        target_video_dir.mkdir(parents=True, exist_ok=True)
        for name in ANNOTATED_VIDEOS:
            shutil.copy2(video_dir / name, target_video_dir / name)
        original_notes = video_dir / "時間.txt"
        if original_notes.is_file():
            shutil.copy2(
                original_notes,
                package_root / "data/annotations/auto_trigger_three_videos_original_time_notes.txt",
            )

        for script_name in ("setup_mac_auto_trigger.sh", "run_mac_auto_trigger.sh"):
            (package_root / "scripts" / script_name).chmod(0o755)
        manifest = {
            "schema_version": 1,
            "release_status": "auto_trigger_tuning_candidate",
            "field_accepted": False,
            "model_retraining_required": False,
            "source_commit": commit,
            "source_branch": branch,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": "knee42-transformer-v12",
            "class_count": 42,
            "annotated_video_count": len(ANNOTATED_VIDEOS),
            "files": _package_file_hashes(package_root),
        }
        (package_root / PACKAGE_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        validate_release_tree(package_root)

        archive.unlink(missing_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path(package_root.name) / path.relative_to(package_root)).as_posix())

    verify_archive(archive)
    digest = sha256_file(archive)
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--verify", type=Path, help="verify an existing ZIP instead of building")
    args = parser.parse_args(argv)
    if args.verify:
        manifest = verify_archive(args.verify)
        print(f"verified {args.verify}: {manifest['source_commit']} ({manifest['class_count']} classes)")
        return 0
    if args.video_dir is None:
        parser.error("--video-dir is required when building")
    archive, sidecar = build_package(
        args.source_root,
        args.video_dir,
        args.output_dir,
        models_dir=args.models_dir,
        vendor_dir=args.vendor_dir,
    )
    print(archive)
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
