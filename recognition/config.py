from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreviewPaths:
    repo_root: Path
    models_dir: Path
    runtime_bundle_dir: Path
    #: 27-class daily30 bundle; superseded by runtime_bundle_dir but kept runnable.
    legacy_bundle_dir: Path
    results_dir: Path
    app_config_path: Path

    @property
    def hand_model(self) -> Path:
        return self.models_dir / "hand_landmarker.task"

    @property
    def pose_model(self) -> Path:
        return self.models_dir / "pose_landmarker.task"


RuntimePaths = PreviewPaths


def _environment_path(name: str, fallback: Path) -> Path:
    configured = os.environ.get(name, "").strip()
    return Path(configured).expanduser().resolve() if configured else fallback.resolve()


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def preview_paths() -> PreviewPaths:
    if _is_frozen():
        executable_dir = Path(sys.executable).resolve().parent
        bundle_root = Path(getattr(sys, "_MEIPASS")).resolve()
        resource_root = bundle_root / "resources"
        repo_root = executable_dir
        models_default = resource_root / "models"
        runtime_default = resource_root / "artifacts" / "realtime" / "best_current"
        legacy_default = resource_root / "artifacts" / "legacy" / "daily30_27class"
        results_default = executable_dir / "logs"
        app_config_default = executable_dir / "app_config.json"
    else:
        repo_root = Path(__file__).resolve().parents[1]
        models_default = repo_root / "models"
        runtime_default = repo_root / "artifacts" / "realtime" / "best_current"
        legacy_default = repo_root / "artifacts" / "legacy" / "daily30_27class"
        results_default = repo_root / "data" / "results"
        app_config_default = repo_root / "app_config.json"

    return PreviewPaths(
        repo_root=repo_root,
        models_dir=_environment_path("SLR_MODELS_DIR", models_default),
        runtime_bundle_dir=_environment_path(
            "SLR_RUNTIME_BUNDLE_DIR", runtime_default
        ),
        legacy_bundle_dir=_environment_path(
            "SLR_LEGACY_BUNDLE_DIR", legacy_default
        ),
        results_dir=_environment_path("SLR_RESULTS_DIR", results_default),
        app_config_path=_environment_path("SLR_APP_CONFIG", app_config_default),
    )
