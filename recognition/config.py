from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreviewPaths:
    repo_root: Path
    models_dir: Path
    runtime_bundle_dir: Path
    results_dir: Path

    @property
    def hand_model(self) -> Path:
        return self.models_dir / "hand_landmarker.task"

    @property
    def pose_model(self) -> Path:
        return self.models_dir / "pose_landmarker.task"


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def preview_paths() -> PreviewPaths:
    repo_root = _resolve_repo_root()
    models_dir = Path(os.environ.get("SLR_MODELS_DIR", repo_root / "models")).resolve()
    runtime_bundle_dir = Path(
        os.environ.get("SLR_RUNTIME_BUNDLE_DIR", repo_root / "artifacts" / "realtime" / "best_current")
    ).resolve()
    results_dir = Path(os.environ.get("SLR_RESULTS_DIR", repo_root / "data" / "results")).resolve()
    return PreviewPaths(
        repo_root=repo_root,
        models_dir=models_dir,
        runtime_bundle_dir=runtime_bundle_dir,
        results_dir=results_dir,
    )
