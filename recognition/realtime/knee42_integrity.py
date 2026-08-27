"""Strict, immutable integrity contracts for Knee42 v13.1 releases."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
MANIFEST_LINE_PATTERN = re.compile(r"([0-9A-Fa-f]{64})[ \t]+(\*?)(.+)")
CANONICAL_RELEASE_SPEC_SHA256 = (
    "b9559421fc90f37a6e42d8fefa488ca59ff4f3af4e53c190d461ba127322623c"
)
DEFAULT_RELEASE_SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "packaging"
    / "knee42_ivcam"
    / "release_spec.json"
)
VERIFIER_REQUIRED_RELEASE_PATHS = frozenset(
    {
        "VERSION_MANIFEST.json",
        "requirements-windows-runtime.lock.txt",
        "packaging/knee42_ivcam/release_spec.json",
        "auto_trigger_knee_ivcam_local.json",
        "auto_trigger_provenance.json",
        "recognition/__init__.py",
        "recognition/inference/__init__.py",
        "recognition/inference/daily30_sentence_model_utils.py",
        "recognition/realtime/__init__.py",
        "recognition/realtime/auto_trigger.py",
        "recognition/realtime/knee42_capture.py",
        "recognition/realtime/knee42_clock.py",
        "recognition/realtime/knee42_controllers.py",
        "recognition/realtime/knee42_display.py",
        "recognition/realtime/knee42_golden.py",
        "recognition/realtime/knee42_integrity.py",
        "recognition/realtime/knee42_ivcam.py",
        "recognition/realtime/knee42_orientation.py",
        "recognition/realtime/knee42_preprocessing.py",
        "recognition/realtime/knee42_session_recording.py",
        "recognition/realtime/probability_reporting.py",
    }
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
    default_model_version: str
    component_manifest_name: str
    label_count: int
    input_shape: tuple[int, int, int]
    artifact_names: Mapping[str, str]
    assets: Mapping[str, AssetSpec]
    model_files: Mapping[str, str]
    required_release_root: tuple[str, ...]
    required_model_layout: tuple[str, ...]
    license_identifiers: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(
            self,
            "artifact_names",
            MappingProxyType(dict(self.artifact_names)),
        )
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))
        object.__setattr__(
            self,
            "model_files",
            MappingProxyType(dict(self.model_files)),
        )
        object.__setattr__(
            self,
            "required_release_root",
            tuple(self.required_release_root),
        )
        object.__setattr__(
            self,
            "required_model_layout",
            tuple(self.required_model_layout),
        )
        object.__setattr__(
            self,
            "license_identifiers",
            MappingProxyType(dict(self.license_identifiers)),
        )

@dataclass(frozen=True)
class ComponentManifest:
    source_sha256: str
    component_id: str
    model_version: str
    label_count: int
    input_shape: tuple[int, int, int]
    runtime_config_sha256: str
    selection_ledger_sha256: str
    payload_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(
            self,
            "payload_sha256",
            MappingProxyType(dict(self.payload_sha256)),
        )


@dataclass(frozen=True)
class VerifiedRelease:
    root: Path
    release_version: str
    app_version: str
    component_id: str
    model_version: str
    model_component_manifest_sha256: str
    label_count: int
    input_shape: tuple[int, int, int]
    source_commit: str
    dependency_lock_sha256: str
    root_manifest_sha256: str
    file_hashes: Mapping[str, str]
    authenticated_files: Mapping[str, bytes] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(
            self,
            "file_hashes",
            MappingProxyType(dict(self.file_hashes)),
        )
        object.__setattr__(
            self,
            "authenticated_files",
            MappingProxyType(
                {
                    str(relative): bytes(payload)
                    for relative, payload in self.authenticated_files.items()
                }
            ),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise IntegrityError(f"non-finite JSON value is forbidden: {value}")


def parse_json_object_bytes(
    raw_bytes: bytes,
    *,
    description: str,
) -> dict[str, Any]:
    """Strictly parse one already-captured UTF-8 JSON object snapshot."""
    try:
        payload = json.loads(
            bytes(raw_bytes).decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except IntegrityError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise IntegrityError(f"cannot parse {description}: {exc}") from exc
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


def require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    description: str,
) -> None:
    """Public exact-schema validator for authenticated JSON snapshots."""
    _require_exact_fields(payload, expected, description=description)


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


def require_sha256(value: Any, *, description: str) -> str:
    """Public lowercase SHA-256 field validator."""
    return _require_sha256(value, description=description)


def read_authenticated_bytes(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> bytes:
    """Read once, hash those exact immutable bytes, and return that snapshot."""
    wanted = _require_sha256(expected_sha256, description=f"{description} SHA-256")
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read {description} {source}: {exc}") from exc
    actual = hashlib.sha256(raw_bytes).hexdigest()
    if actual != wanted:
        raise IntegrityError(
            f"{description} SHA-256 mismatch for {source.name}: "
            f"expected {wanted}, actual {actual}"
        )
    return raw_bytes


def authenticated_release_bytes(
    release: VerifiedRelease,
    relative_path: str,
    *,
    description: str,
) -> bytes:
    """Return one immutable release snapshot already bound to the root anchor."""
    if not isinstance(release, VerifiedRelease):
        raise IntegrityError("trusted release must be a VerifiedRelease")
    relative = _normalize_relative_path(
        relative_path,
        description=f"{description} relative path",
    )
    expected = (
        release.root_manifest_sha256
        if relative == "integrity_manifest.sha256"
        else release.file_hashes.get(relative)
    )
    if expected is None:
        raise IntegrityError(f"trusted release missing {description}: {relative}")
    raw_bytes = release.authenticated_files.get(relative)
    if raw_bytes is None:
        raise IntegrityError(
            f"trusted release missing authenticated bytes for {description}: {relative}"
        )
    snapshot = bytes(raw_bytes)
    actual = hashlib.sha256(snapshot).hexdigest()
    if actual != expected:
        raise IntegrityError(
            f"authenticated {description} SHA-256 mismatch: "
            f"expected {expected}, actual {actual}"
        )
    return snapshot


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
    try:
        raw_bytes = spec_path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read release spec {spec_path}: {exc}") from exc
    payload = parse_json_object_bytes(raw_bytes, description="release spec")
    source_sha256 = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()
    expected_fields = {
        "schema_version",
        "release_version",
        "app_version",
        "default_model_version",
        "component_manifest_name",
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
    default_model_version = _require_string(
        payload["default_model_version"],
        description="release spec default_model_version",
    )
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+-v[0-9]+\.[0-9]+", release_version) is None:
        raise IntegrityError("release spec release_version has an invalid value")
    if re.fullmatch(r"v[0-9]+\.[0-9]+", app_version) is None:
        raise IntegrityError("release spec app_version has an invalid value")
    if re.fullmatch(r"v[0-9]+", default_model_version) is None:
        raise IntegrityError("release spec default_model_version has an invalid value")
    component_manifest_name = _normalize_relative_path(
        payload["component_manifest_name"],
        description="release spec component_manifest_name",
    )
    if "/" in component_manifest_name or component_manifest_name != "component_manifest.json":
        raise IntegrityError(
            "release spec component_manifest_name must be component_manifest.json"
        )

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
    if component_manifest_name not in required_model_layout:
        raise IntegrityError(
            "release spec required model layout omits component manifest: "
            f"{component_manifest_name}"
        )
    missing_verifier_paths = sorted(
        VERIFIER_REQUIRED_RELEASE_PATHS - set(required_release_root)
    )
    if missing_verifier_paths:
        raise IntegrityError(
            "release spec required release-root layout omits verifier-required path(s): "
            f"{missing_verifier_paths}"
        )
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
        default_model_version=default_model_version,
        component_manifest_name=component_manifest_name,
        label_count=label_count,
        input_shape=input_shape,
        artifact_names=artifacts,
        assets=MappingProxyType(assets),
        model_files=MappingProxyType(model_files),
        required_release_root=required_release_root,
        required_model_layout=required_model_layout,
        license_identifiers=licenses,
    )


def load_component_manifest(
    path: Path,
    *,
    expected_sha256: str,
    spec: ReleaseSpec,
) -> ComponentManifest:
    """Authenticate raw component bytes before parsing the strict component schema."""
    raw_bytes = read_authenticated_bytes(
        path,
        expected_sha256=expected_sha256,
        description="component manifest",
    )
    return load_component_manifest_bytes(
        raw_bytes,
        expected_sha256=expected_sha256,
        spec=spec,
    )


def load_component_manifest_bytes(
    raw_bytes: bytes,
    *,
    expected_sha256: str,
    spec: ReleaseSpec,
) -> ComponentManifest:
    """Authenticate and strictly parse one captured component-manifest snapshot."""
    trusted_hash = _require_sha256(
        expected_sha256,
        description="expected component manifest SHA-256",
    )
    snapshot = bytes(raw_bytes)
    actual_hash = hashlib.sha256(snapshot).hexdigest()
    if actual_hash != trusted_hash:
        raise IntegrityError(
            "component manifest SHA-256 mismatch: "
            f"expected {trusted_hash}, actual {actual_hash}"
        )

    payload = parse_json_object_bytes(snapshot, description="component manifest")
    _require_exact_fields(
        payload,
        {
            "schema_version",
            "component_id",
            "model_version",
            "label_count",
            "input_shape",
            "runtime_config_sha256",
            "selection_ledger_sha256",
            "payload_sha256",
        },
        description="component manifest",
    )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise IntegrityError("component manifest schema_version must be integer 1")
    component_id = _require_string(
        payload["component_id"], description="component manifest component_id"
    )
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", component_id) is None:
        raise IntegrityError("component manifest component_id has an invalid value")
    model_version = _require_string(
        payload["model_version"], description="component manifest model_version"
    )
    if re.fullmatch(r"v[0-9]+", model_version) is None:
        raise IntegrityError("component manifest model_version has an invalid value")
    if type(payload["label_count"]) is not int or payload["label_count"] != spec.label_count:
        raise IntegrityError(
            "component manifest label_count mismatch: "
            f"expected {spec.label_count}, got {payload['label_count']!r}"
        )
    raw_shape = payload["input_shape"]
    if (
        type(raw_shape) is not list
        or len(raw_shape) != 3
        or any(type(item) is not int for item in raw_shape)
        or tuple(raw_shape) != spec.input_shape
    ):
        raise IntegrityError(
            "component manifest input_shape mismatch: "
            f"expected {list(spec.input_shape)}, got {raw_shape!r}"
        )

    raw_hashes = _require_mapping(
        payload["payload_sha256"], description="component manifest payload_sha256"
    )
    expected_payload_names = set(spec.required_model_layout) - {
        spec.component_manifest_name,
        "integrity_manifest.sha256",
    }
    actual_payload_names: set[str] = set()
    payload_hashes: dict[str, str] = {}
    for raw_name, raw_digest in raw_hashes.items():
        name = _normalize_relative_path(
            raw_name, description="component manifest payload"
        )
        if "/" in name:
            raise IntegrityError(
                f"component manifest payload name must be a basename: {raw_name}"
            )
        actual_payload_names.add(name)
        payload_hashes[name] = _require_sha256(
            raw_digest,
            description=f"component manifest payload SHA-256 {name}",
        )
    missing = sorted(expected_payload_names - actual_payload_names)
    unexpected = sorted(actual_payload_names - expected_payload_names)
    if missing:
        raise IntegrityError(f"component manifest payload missing path(s): {missing}")
    if unexpected:
        raise IntegrityError(f"component manifest payload unexpected path(s): {unexpected}")
    runtime_config_sha256 = _require_sha256(
        payload["runtime_config_sha256"],
        description="component manifest runtime_config_sha256",
    )
    if runtime_config_sha256 != payload_hashes["runtime_config.json"]:
        raise IntegrityError(
            "component manifest runtime_config_sha256 must equal "
            "payload_sha256['runtime_config.json']"
        )
    selection_ledger_sha256 = _require_sha256(
        payload["selection_ledger_sha256"],
        description="component manifest selection_ledger_sha256",
    )
    if selection_ledger_sha256 != payload_hashes["selection_ledger.json"]:
        raise IntegrityError(
            "component manifest selection_ledger_sha256 must equal "
            "payload_sha256['selection_ledger.json']"
        )

    return ComponentManifest(
        source_sha256=actual_hash,
        component_id=component_id,
        model_version=model_version,
        label_count=spec.label_count,
        input_shape=spec.input_shape,
        runtime_config_sha256=runtime_config_sha256,
        selection_ledger_sha256=selection_ledger_sha256,
        payload_sha256=payload_hashes,
    )


def parse_sha256_manifest_bytes(
    raw_bytes: bytes,
    *,
    description: str = "SHA-256 manifest",
) -> Mapping[str, str]:
    """Parse one captured sha256sum snapshot and reject ambiguous paths."""
    try:
        lines = bytes(raw_bytes).decode("ascii").splitlines()
    except UnicodeError as exc:
        raise IntegrityError(f"cannot decode {description}: {exc}") from exc

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
        if match.group(2) or "*" in raw_path:
            raise IntegrityError(
                f"ambiguous GNU '*' manifest syntax or filename at line {line_number}"
            )
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
        raise IntegrityError(f"{description} is empty")
    return MappingProxyType(hashes)


def parse_sha256_manifest(path: Path) -> Mapping[str, str]:
    """Read and parse a sha256sum manifest once."""
    manifest_path = Path(path)
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read SHA-256 manifest {manifest_path}: {exc}") from exc
    return parse_sha256_manifest_bytes(
        raw_bytes,
        description=f"SHA-256 manifest {manifest_path}",
    )


def _relative_files(root: Path) -> set[str]:
    files: set[str] = set()
    pending: list[tuple[Path, str]] = [(root, "")]
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise IntegrityError(
                            f"cannot inspect release path {relative}: {exc}"
                        ) from exc
                    is_reparse = bool(
                        getattr(metadata, "st_file_attributes", 0)
                        & reparse_attribute
                    )
                    if stat.S_ISLNK(metadata.st_mode) or is_reparse:
                        raise IntegrityError(
                            f"unexpected symbolic link or reparse point: {relative}"
                        )
                    if stat.S_ISDIR(metadata.st_mode):
                        pending.append((Path(entry.path), relative))
                    elif stat.S_ISREG(metadata.st_mode):
                        files.add(relative)
                    else:
                        raise IntegrityError(f"unexpected non-regular path: {relative}")
        except IntegrityError:
            raise
        except OSError as exc:
            location = prefix or "."
            raise IntegrityError(
                f"cannot enumerate release path {location}: {exc}"
            ) from exc
    return files


def _validate_version_manifest_bytes(
    raw_bytes: bytes,
    spec: ReleaseSpec,
    component: ComponentManifest,
) -> dict[str, Any]:
    payload = parse_json_object_bytes(
        raw_bytes,
        description="VERSION_MANIFEST.json",
    )
    expected_fields = {
        "release_version",
        "app_version",
        "component_id",
        "model_version",
        "model_component_manifest_sha256",
        "label_count",
        "input_shape",
        "source_commit",
        "dependency_lock_sha256",
    }
    _require_exact_fields(payload, expected_fields, description="VERSION_MANIFEST.json")
    input_shape = payload["input_shape"]
    if (
        type(input_shape) is not list
        or len(input_shape) != len(spec.input_shape)
        or any(type(item) is not int for item in input_shape)
    ):
        raise IntegrityError(
            "VERSION_MANIFEST.json input_shape must contain exactly three integers"
        )
    expected_values: dict[str, Any] = {
        "release_version": spec.release_version,
        "app_version": spec.app_version,
        "component_id": component.component_id,
        "model_version": component.model_version,
        "model_component_manifest_sha256": component.source_sha256,
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
    expected_root_manifest_sha256: str,
    spec: ReleaseSpec | None = None,
) -> VerifiedRelease:
    """Verify every listed release file, reject extras, and validate version lineage."""
    trusted_manifest_hash = _require_sha256(
        expected_root_manifest_sha256,
        description="expected_root_manifest_sha256",
    )
    release_root = Path(root).resolve()
    if not release_root.is_dir():
        raise IntegrityError(f"release root missing or not a directory: {release_root}")
    actual = _relative_files(release_root)
    manifest_path = release_root / "integrity_manifest.sha256"
    if not manifest_path.is_file():
        raise IntegrityError("missing path: integrity_manifest.sha256")
    manifest_bytes = read_authenticated_bytes(
        manifest_path,
        expected_sha256=trusted_manifest_hash,
        description="root integrity manifest",
    )
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    if spec is None:
        trusted_spec = load_release_spec(DEFAULT_RELEASE_SPEC_PATH)
        if trusted_spec.source_sha256 != CANONICAL_RELEASE_SPEC_SHA256:
            raise IntegrityError(
                "canonical release spec SHA-256 mismatch: "
                f"expected {CANONICAL_RELEASE_SPEC_SHA256}, "
                f"actual {trusted_spec.source_sha256}"
            )
    else:
        trusted_spec = spec
    hashes = parse_sha256_manifest_bytes(
        manifest_bytes,
        description="root integrity manifest integrity_manifest.sha256",
    )
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

    expected = set(hashes) | {"integrity_manifest.sha256"}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise IntegrityError(f"missing file path(s): {missing}")
    if unexpected:
        raise IntegrityError(f"unexpected file path(s): {unexpected}")

    authenticated_files: dict[str, bytes] = {
        "integrity_manifest.sha256": manifest_bytes,
    }
    for relative, wanted in hashes.items():
        path = release_root.joinpath(*relative.split("/"))
        authenticated_files[relative] = read_authenticated_bytes(
            path,
            expected_sha256=wanted,
            description="release path",
        )

    packaged_spec_relative = "packaging/knee42_ivcam/release_spec.json"
    actual_packaged_spec_hash = hashlib.sha256(
        authenticated_files[packaged_spec_relative].replace(b"\r\n", b"\n")
    ).hexdigest()
    if actual_packaged_spec_hash != trusted_spec.source_sha256:
        raise IntegrityError(
            f"canonical SHA-256 mismatch for {packaged_spec_relative}: "
            f"expected {trusted_spec.source_sha256}, actual {actual_packaged_spec_hash}"
        )

    canonical_hashes = {
        f"model/{trusted_spec.assets[name].filename}": trusted_spec.assets[name].sha256
        for name in ("hand_landmarker_task", "pose_landmarker_task")
    }
    for relative, wanted in canonical_hashes.items():
        declared = hashes.get(relative)
        if declared != wanted:
            raise IntegrityError(
                f"canonical SHA-256 mismatch for {relative}: "
                f"expected {wanted}, manifest declares {declared}"
            )

    component_relative = f"model/{trusted_spec.component_manifest_name}"
    component = load_component_manifest_bytes(
        authenticated_files[component_relative],
        expected_sha256=hashes[component_relative],
        spec=trusted_spec,
    )
    for name, wanted in component.payload_sha256.items():
        relative = f"model/{name}"
        declared = hashes.get(relative)
        if declared != wanted:
            raise IntegrityError(
                f"component payload SHA-256 mismatch for {relative}: "
                f"component declares {wanted}, root manifest declares {declared}"
            )
    internal_manifest = parse_sha256_manifest_bytes(
        authenticated_files["model/integrity_manifest.sha256"],
        description="component integrity manifest",
    )
    if dict(internal_manifest) != dict(component.payload_sha256):
        missing = sorted(set(component.payload_sha256) - set(internal_manifest))
        unexpected = sorted(set(internal_manifest) - set(component.payload_sha256))
        mismatched = sorted(
            name
            for name in set(component.payload_sha256) & set(internal_manifest)
            if component.payload_sha256[name] != internal_manifest[name]
        )
        raise IntegrityError(
            "component payload SHA-256 mismatch with internal integrity manifest: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    version = _validate_version_manifest_bytes(
        authenticated_files["VERSION_MANIFEST.json"],
        trusted_spec,
        component,
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
        component_id=component.component_id,
        model_version=component.model_version,
        model_component_manifest_sha256=component.source_sha256,
        label_count=trusted_spec.label_count,
        input_shape=trusted_spec.input_shape,
        source_commit=version["source_commit"],
        dependency_lock_sha256=version["dependency_lock_sha256"],
        root_manifest_sha256=actual_manifest_hash,
        file_hashes=MappingProxyType(dict(hashes)),
        authenticated_files=MappingProxyType(authenticated_files),
    )
