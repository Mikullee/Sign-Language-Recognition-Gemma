"""Strict, immutable integrity contracts for Knee42 v13.1 releases."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
MANIFEST_LINE_PATTERN = re.compile(r"([0-9A-Fa-f]{64})[ \t]+(\*?)(.+)")
CANONICAL_RELEASE_SPEC_SHA256 = (
    "d6a9a3d9e9e6eb8932f7b3012b762a41b62be43ced2acb83edee9de9876c122b"
)
DEFAULT_RELEASE_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "packaging"
    / "knee42_ivcam"
    / "release_spec.json"
)


class IntegrityError(RuntimeError):
    """Raised when a release contract or checked artifact is invalid."""


@dataclass(frozen=True)
class AssetSpec:
    filename: str
    url: str
    sha256: str
    license_id: str


@dataclass(frozen=True)
class ReleaseSpec:
    source_sha256: str
    release_version: str
    app_version: str
    model_version: str
    label_count: int
    input_shape: tuple[int, int, int]
    artifact_names: Mapping[str, str]
    assets: Mapping[str, AssetSpec]
    model_files: Mapping[str, str]
    required_release_root: tuple[str, ...]
    required_model_layout: tuple[str, ...]
    license_identifiers: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedRelease:
    root: Path
    release_version: str
    app_version: str
    model_version: str
    label_count: int
    input_shape: tuple[int, int, int]
    source_commit: str
    dependency_lock_sha256: str
    root_manifest_sha256: str
    file_hashes: Mapping[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf_normalized_file(path: Path) -> str:
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except IntegrityError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise IntegrityError(f"cannot read {description} {Path(path).name}: {exc}") from exc
    if type(payload) is not dict:
        raise IntegrityError(f"{description} must be a JSON object")
    return payload


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    description: str,
) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing:
        raise IntegrityError(f"{description} missing field(s): {missing}")
    if unknown:
        raise IntegrityError(f"{description} unknown field(s): {unknown}")


def _require_string(value: Any, *, description: str) -> str:
    if type(value) is not str or not value:
        raise IntegrityError(f"{description} must be a non-empty string")
    return value


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise IntegrityError(f"{description} must be a JSON object")
    return value


def _require_sha256(value: Any, *, description: str) -> str:
    text = _require_string(value, description=description)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise IntegrityError(f"{description} is invalid: {text!r}")
    return text


def _normalize_relative_path(value: Any, *, description: str) -> str:
    text = _require_string(value, description=description)
    if any(ord(character) < 32 for character in text):
        raise IntegrityError(f"{description} path contains a control character: {text!r}")
    normalized = text.replace("\\", "/")
    if (
        normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized) is not None
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or any(":" in part for part in normalized.split("/"))
    ):
        raise IntegrityError(f"{description} path is unsafe: {text}")
    return normalized


def _load_string_mapping(
    value: Any,
    *,
    description: str,
    required_keys: set[str] | None = None,
) -> Mapping[str, str]:
    payload = _require_mapping(value, description=description)
    if required_keys is not None:
        _require_exact_fields(payload, required_keys, description=description)
    result = {
        _require_string(key, description=f"{description} key"): _require_string(
            item, description=f"{description}.{key}"
        )
        for key, item in payload.items()
    }
    return MappingProxyType(result)


def _load_layout(value: Any, *, description: str, basenames_only: bool) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise IntegrityError(f"{description} must be a non-empty list")
    result = tuple(
        _normalize_relative_path(item, description="unsafe layout") for item in value
    )
    if len(set(result)) != len(result):
        raise IntegrityError(f"{description} contains a duplicate path")
    if basenames_only and any("/" in item for item in result):
        raise IntegrityError(f"{description} entries must be model-directory basenames")
    return result


def load_release_spec(path: Path) -> ReleaseSpec:
    """Load the canonical release spec with strict schema and value validation."""
    spec_path = Path(path)
    payload = _read_json_object(spec_path, description="release spec")
    try:
        source_sha256 = sha256_file(spec_path)
    except OSError as exc:
        raise IntegrityError(f"cannot hash release spec {spec_path}: {exc}") from exc
    expected_fields = {
        "schema_version",
        "release_version",
        "app_version",
        "model_version",
        "label_count",
        "input_shape",
        "artifact_names",
        "assets",
        "model_files",
        "required_layouts",
        "license_identifiers",
    }
    _require_exact_fields(payload, expected_fields, description="release spec")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise IntegrityError("release spec schema_version must be integer 1")

    release_version = _require_string(
        payload["release_version"], description="release spec release_version"
    )
    app_version = _require_string(payload["app_version"], description="release spec app_version")
    model_version = _require_string(
        payload["model_version"], description="release spec model_version"
    )
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+-v[0-9]+\.[0-9]+", release_version) is None:
        raise IntegrityError("release spec release_version has an invalid value")
    if re.fullmatch(r"v[0-9]+\.[0-9]+", app_version) is None:
        raise IntegrityError("release spec app_version has an invalid value")
    if re.fullmatch(r"v[0-9]+", model_version) is None:
        raise IntegrityError("release spec model_version has an invalid value")

    label_count = payload["label_count"]
    if type(label_count) is not int or label_count <= 0:
        raise IntegrityError("release spec label_count must be a positive integer")
    shape = payload["input_shape"]
    if (
        type(shape) is not list
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise IntegrityError("release spec input_shape must contain three positive integers")
    input_shape = (shape[0], shape[1], shape[2])

    artifacts = _load_string_mapping(
        payload["artifact_names"],
        description="release spec artifact_names",
        required_keys={"source_runtime", "windows_x64"},
    )
    for key, filename in artifacts.items():
        if _normalize_relative_path(filename, description="artifact name") != filename:
            raise IntegrityError(f"artifact name must use POSIX separators: {filename}")
        if "/" in filename or not filename.endswith(".zip"):
            raise IntegrityError(f"release spec artifact name {key} must be a ZIP basename")

    raw_assets = _require_mapping(payload["assets"], description="release spec assets")
    _require_exact_fields(
        raw_assets,
        {"model_archive", "hand_landmarker_task", "pose_landmarker_task"},
        description="release spec assets",
    )
    assets: dict[str, AssetSpec] = {}
    for name, raw_asset in raw_assets.items():
        asset = _require_mapping(raw_asset, description=f"release spec asset {name}")
        _require_exact_fields(
            asset,
            {"filename", "url", "sha256", "license_id"},
            description=f"release spec asset {name}",
        )
        filename = _normalize_relative_path(
            asset["filename"], description=f"release spec asset filename {name}"
        )
        if "/" in filename:
            raise IntegrityError(f"release spec asset filename {name} must be a basename")
        url = _require_string(asset["url"], description=f"release spec asset URL {name}")
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.username:
            raise IntegrityError(f"release spec asset URL {name} must be an HTTPS URL")
        digest = _require_sha256(
            asset["sha256"], description=f"release spec asset SHA-256 {name}"
        )
        license_id = _require_string(
            asset["license_id"], description=f"release spec asset license_id {name}"
        )
        if LICENSE_PATTERN.fullmatch(license_id) is None:
            raise IntegrityError(f"release spec asset license_id {name} is invalid")
        assets[name] = AssetSpec(filename, url, digest, license_id)

    raw_model_files = _require_mapping(
        payload["model_files"], description="release spec model_files"
    )
    if not raw_model_files:
        raise IntegrityError("release spec model_files must not be empty")
    model_files: dict[str, str] = {}
    for filename, digest in raw_model_files.items():
        normalized = _normalize_relative_path(filename, description="model file")
        if "/" in normalized:
            raise IntegrityError(f"release spec model file must be a basename: {filename}")
        model_files[normalized] = _require_sha256(
            digest, description=f"release spec model file SHA-256 {normalized}"
        )

    raw_layouts = _require_mapping(
        payload["required_layouts"], description="release spec required_layouts"
    )
    _require_exact_fields(
        raw_layouts,
        {"release_root", "model"},
        description="release spec required_layouts",
    )
    required_release_root = _load_layout(
        raw_layouts["release_root"],
        description="release spec required release-root layout",
        basenames_only=False,
    )
    required_model_layout = _load_layout(
        raw_layouts["model"],
        description="release spec required model layout",
        basenames_only=True,
    )
    if "VERSION_MANIFEST.json" not in required_release_root:
        raise IntegrityError("release spec required release-root layout lacks VERSION_MANIFEST.json")
    if not set(model_files).issubset(required_model_layout):
        missing = sorted(set(model_files) - set(required_model_layout))
        raise IntegrityError(f"release spec required model layout omits model file(s): {missing}")
    for asset_name in ("hand_landmarker_task", "pose_landmarker_task"):
        if assets[asset_name].filename not in required_model_layout:
            raise IntegrityError(
                "release spec required model layout omits asset: "
                f"{assets[asset_name].filename}"
            )

    licenses = _load_string_mapping(
        payload["license_identifiers"],
        description="release spec license_identifiers",
        required_keys={"application", "knee42_model", "mediapipe_tasks"},
    )
    for name, license_id in licenses.items():
        if LICENSE_PATTERN.fullmatch(license_id) is None:
            raise IntegrityError(f"release spec license identifier {name} is invalid")
    if assets["hand_landmarker_task"].license_id != licenses["mediapipe_tasks"]:
        raise IntegrityError("release spec hand task license identifier is inconsistent")
    if assets["pose_landmarker_task"].license_id != licenses["mediapipe_tasks"]:
        raise IntegrityError("release spec pose task license identifier is inconsistent")
    if assets["model_archive"].license_id != licenses["knee42_model"]:
        raise IntegrityError("release spec model archive license identifier is inconsistent")

    return ReleaseSpec(
        source_sha256=source_sha256,
        release_version=release_version,
        app_version=app_version,
        model_version=model_version,
        label_count=label_count,
        input_shape=input_shape,
        artifact_names=artifacts,
        assets=MappingProxyType(assets),
        model_files=MappingProxyType(model_files),
        required_release_root=required_release_root,
        required_model_layout=required_model_layout,
        license_identifiers=licenses,
    )


def parse_sha256_manifest(path: Path) -> Mapping[str, str]:
    """Parse a sha256sum manifest and reject unsafe or ambiguous paths."""
    manifest_path = Path(path)
    try:
        lines = manifest_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise IntegrityError(f"cannot read SHA-256 manifest {manifest_path}: {exc}") from exc

    hashes: dict[str, str] = {}
    casefolded_paths: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        match = MANIFEST_LINE_PATTERN.fullmatch(line)
        if match is None or not match.group(3).strip():
            raise IntegrityError(f"malformed SHA-256 manifest line {line_number}")
        digest = match.group(1).lower()
        raw_path = match.group(3)
        if raw_path != raw_path.strip():
            raise IntegrityError(f"malformed SHA-256 manifest path at line {line_number}")
        normalized = _normalize_relative_path(raw_path, description="unsafe integrity")
        folded = normalized.casefold()
        if folded in casefolded_paths:
            raise IntegrityError(
                "duplicate integrity path at line "
                f"{line_number}: {normalized} (already {casefolded_paths[folded]})"
            )
        casefolded_paths[folded] = normalized
        hashes[normalized] = digest
    if not hashes:
        raise IntegrityError(f"SHA-256 manifest is empty: {manifest_path}")
    return MappingProxyType(hashes)


def _relative_files(root: Path) -> set[str]:
    files: set[str] = set()
    try:
        paths = list(root.rglob("*"))
    except OSError as exc:
        raise IntegrityError(f"cannot enumerate release root {root}: {exc}") from exc
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise IntegrityError(f"unexpected symbolic link: {relative}")
        if path.is_file():
            files.add(relative)
    return files


def _validate_version_manifest(path: Path, spec: ReleaseSpec) -> dict[str, Any]:
    payload = _read_json_object(path, description="VERSION_MANIFEST.json")
    expected_fields = {
        "release_version",
        "app_version",
        "model_version",
        "label_count",
        "input_shape",
        "source_commit",
        "dependency_lock_sha256",
    }
    _require_exact_fields(payload, expected_fields, description="VERSION_MANIFEST.json")
    expected_values: dict[str, Any] = {
        "release_version": spec.release_version,
        "app_version": spec.app_version,
        "model_version": spec.model_version,
        "label_count": spec.label_count,
        "input_shape": list(spec.input_shape),
    }
    for field, expected in expected_values.items():
        actual = payload[field]
        if type(actual) is not type(expected) or actual != expected:
            raise IntegrityError(
                f"VERSION_MANIFEST.json {field} mismatch: expected {expected!r}, got {actual!r}"
            )
    source_commit = payload["source_commit"]
    if type(source_commit) is not str or COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise IntegrityError("VERSION_MANIFEST.json source_commit must be 40 lowercase hex characters")
    lock_hash = payload["dependency_lock_sha256"]
    if type(lock_hash) is not str or SHA256_PATTERN.fullmatch(lock_hash) is None:
        raise IntegrityError(
            "VERSION_MANIFEST.json dependency_lock_sha256 must be 64 lowercase hex characters"
        )
    return payload


def verify_release_root(
    root: Path,
    *,
    spec: ReleaseSpec | None = None,
) -> VerifiedRelease:
    """Verify every listed release file, reject extras, and validate version lineage."""
    release_root = Path(root).resolve()
    if not release_root.is_dir():
        raise IntegrityError(f"release root missing or not a directory: {release_root}")
    manifest_path = release_root / "integrity_manifest.sha256"
    if not manifest_path.is_file():
        raise IntegrityError("missing path: integrity_manifest.sha256")

    if spec is None:
        try:
            actual_spec_hash = _sha256_lf_normalized_file(DEFAULT_RELEASE_SPEC_PATH)
        except OSError as exc:
            raise IntegrityError(
                f"cannot hash canonical release spec {DEFAULT_RELEASE_SPEC_PATH}: {exc}"
            ) from exc
        if actual_spec_hash != CANONICAL_RELEASE_SPEC_SHA256:
            raise IntegrityError(
                "canonical release spec SHA-256 mismatch: "
                f"expected {CANONICAL_RELEASE_SPEC_SHA256}, actual {actual_spec_hash}"
            )
        trusted_spec = load_release_spec(DEFAULT_RELEASE_SPEC_PATH)
    else:
        trusted_spec = spec
    hashes = parse_sha256_manifest(manifest_path)
    if "integrity_manifest.sha256" in hashes:
        raise IntegrityError("integrity manifest must not list itself: integrity_manifest.sha256")

    expected_manifest_paths = set(trusted_spec.required_release_root)
    expected_manifest_paths.update(
        f"model/{name}" for name in trusted_spec.required_model_layout
    )
    manifest_paths = set(hashes)
    missing_manifest_paths = sorted(expected_manifest_paths - manifest_paths)
    unexpected_manifest_paths = sorted(manifest_paths - expected_manifest_paths)
    if missing_manifest_paths:
        raise IntegrityError(
            f"integrity manifest missing required path(s): {missing_manifest_paths}"
        )
    if unexpected_manifest_paths:
        raise IntegrityError(
            f"integrity manifest has unexpected path(s): {unexpected_manifest_paths}"
        )

    canonical_hashes = {
        "packaging/knee42_ivcam/release_spec.json": trusted_spec.source_sha256,
        **{
            f"model/{filename}": digest
            for filename, digest in trusted_spec.model_files.items()
        },
        **{
            f"model/{trusted_spec.assets[name].filename}": trusted_spec.assets[name].sha256
            for name in ("hand_landmarker_task", "pose_landmarker_task")
        },
    }
    for relative, wanted in canonical_hashes.items():
        declared = hashes.get(relative)
        if declared != wanted:
            raise IntegrityError(
                f"canonical SHA-256 mismatch for {relative}: "
                f"expected {wanted}, manifest declares {declared}"
            )

    actual = _relative_files(release_root)
    expected = set(hashes) | {"integrity_manifest.sha256"}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise IntegrityError(f"missing file path(s): {missing}")
    if unexpected:
        raise IntegrityError(f"unexpected file path(s): {unexpected}")

    for relative, wanted in hashes.items():
        path = release_root.joinpath(*relative.split("/"))
        try:
            actual_hash = sha256_file(path)
        except OSError as exc:
            raise IntegrityError(f"cannot hash release path {relative}: {exc}") from exc
        if actual_hash != wanted:
            raise IntegrityError(
                f"SHA-256 mismatch for {relative}: expected {wanted}, actual {actual_hash}"
            )

    version = _validate_version_manifest(
        release_root / "VERSION_MANIFEST.json",
        trusted_spec,
    )
    lock_relative = "requirements-windows-runtime.lock.txt"
    if version["dependency_lock_sha256"] != hashes[lock_relative]:
        raise IntegrityError(
            "VERSION_MANIFEST.json dependency_lock_sha256 mismatch for "
            f"{lock_relative}: expected {hashes[lock_relative]}, "
            f"got {version['dependency_lock_sha256']}"
        )
    return VerifiedRelease(
        root=release_root,
        release_version=trusted_spec.release_version,
        app_version=trusted_spec.app_version,
        model_version=trusted_spec.model_version,
        label_count=trusted_spec.label_count,
        input_shape=trusted_spec.input_shape,
        source_commit=version["source_commit"],
        dependency_lock_sha256=version["dependency_lock_sha256"],
        root_manifest_sha256=sha256_file(manifest_path),
        file_hashes=MappingProxyType(dict(hashes)),
    )
