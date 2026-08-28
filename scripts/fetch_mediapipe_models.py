"""Download the two MediaPipe landmarker models this project was trained with.

The ``.task`` files are not redistributed here: Google publishes them under its
own model-card terms, which are not necessarily the Apache-2.0 licence covering
the MediaPipe source (see THIRD_PARTY_NOTICES.md).  This script fetches them
from Google's official storage instead, so a fresh clone is one command away
from running.

Every download is checked against the SHA-256 recorded in the feature cache the
model was trained on.  That check is the point of the script: the pose model
must be **lite**, not **full**, and picking the wrong one raises no error at
all -- it just quietly moves inference onto a different landmark distribution
than the one the weights were fitted to.

Usage
-----
    python scripts/fetch_mediapipe_models.py
    python scripts/fetch_mediapipe_models.py --dest models --force
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "hand_landmarker.task": {
        "url": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        "note": "hand landmarker, float16",
    },
    "pose_landmarker.task": {
        "url": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        "note": "pose landmarker LITE, float16 -- not full",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Fetch to a temporary file first, so a failure cannot leave a partial model."""
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as staged:
        staging_path = Path(staged.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, staging_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        staging_path.replace(destination)
    except BaseException:
        staging_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true", help="re-download even if the hash matches")
    args = parser.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    failures = 0
    for filename, spec in MODELS.items():
        target = args.dest / filename
        if target.is_file() and not args.force:
            actual = sha256_file(target)
            if actual == spec["sha256"]:
                print(f"{filename}: already present and verified")
                continue
            print(f"{filename}: present but hash differs, re-downloading", file=sys.stderr)

        print(f"{filename}: downloading ({spec['note']})")
        try:
            download(spec["url"], target)
        except (urllib.error.URLError, OSError) as exc:
            print(f"{filename}: download failed: {exc}", file=sys.stderr)
            failures += 1
            continue

        actual = sha256_file(target)
        if actual != spec["sha256"]:
            target.unlink(missing_ok=True)
            print(
                f"{filename}: SHA-256 MISMATCH, file removed\n"
                f"  expected {spec['sha256']}\n"
                f"  actual   {actual}\n"
                f"  Google may have republished this model. Do not use a mismatched file:\n"
                f"  the features would no longer match the ones the weights were trained on.",
                file=sys.stderr,
            )
            failures += 1
            continue
        print(f"{filename}: verified {actual[:16]}...")

    if failures:
        print(f"\n{failures} model(s) unavailable; see models/README.md", file=sys.stderr)
        return 1
    print(f"\nboth models verified in {args.dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
