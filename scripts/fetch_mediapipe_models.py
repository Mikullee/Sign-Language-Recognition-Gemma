"""Download the MediaPipe assets a fresh clone needs, and verify every one.

Two sets, for two different runtimes:

**Python** (``models/``) -- the ``.task`` landmarkers used by the CLI, the
realtime path, and server-side video analysis. Verified against the SHA-256
recorded in the feature cache the weights were trained on. The pose model must
be **lite**, not **full**: using full raises no error anywhere, it just moves
inference onto a different landmark distribution than the one the model was
fitted to.

**Browser** (``webservice/vendor/mediapipe/``) -- the WebAssembly build the web
service's camera mode loads, so MediaPipe runs in the viewer's browser and the
video never reaches the server. Pinned to ``@mediapipe/tasks-vision`` 0.10.35,
the same version as the Python package, confirmed by hash against a known-good
deployment.

None of these are redistributed in this repository: Google publishes them under
its own model-card and package terms rather than the Apache-2.0 licence covering
the MediaPipe source. See THIRD_PARTY_NOTICES.md.

Usage
-----
    python scripts/fetch_mediapipe_models.py              # both sets
    python scripts/fetch_mediapipe_models.py --python     # .task only
    python scripts/fetch_mediapipe_models.py --browser    # web assets only
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


MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
TASKS_VISION_VERSION = "0.10.35"
WEB_BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VISION_VERSION}"

PYTHON_MODELS = {
    "hand_landmarker.task": {
        "url": f"{MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        "note": "hand landmarker, float16",
    },
    "pose_landmarker.task": {
        "url": f"{MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "sha256": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        "note": "pose landmarker LITE, float16 -- not full",
    },
}

BROWSER_ASSETS = {
    "vision_bundle.mjs": {
        "url": f"{WEB_BASE}/vision_bundle.mjs",
        "sha256": "55d7ab624fbb70dcc5adc4ae6d7ea9cfcb569139d3dbfbf2b1deafcb966bc0fe",
        "note": "Tasks Vision ES module",
    },
    "wasm/vision_wasm_internal.js": {
        "url": f"{WEB_BASE}/wasm/vision_wasm_internal.js",
        "sha256": "e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c",
        "note": "SIMD loader",
    },
    "wasm/vision_wasm_internal.wasm": {
        "url": f"{WEB_BASE}/wasm/vision_wasm_internal.wasm",
        "sha256": "6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc",
        "note": "SIMD runtime",
    },
    "wasm/vision_wasm_nosimd_internal.js": {
        "url": f"{WEB_BASE}/wasm/vision_wasm_nosimd_internal.js",
        "sha256": "438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296",
        "note": "non-SIMD loader, for browsers without SIMD",
    },
    "wasm/vision_wasm_nosimd_internal.wasm": {
        "url": f"{WEB_BASE}/wasm/vision_wasm_nosimd_internal.wasm",
        "sha256": "8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31",
        "note": "non-SIMD runtime",
    },
}

#: The browser loads the landmarkers from the same route it loads the wasm from,
#: so the vendor directory needs its own copies rather than a path into models/.
BROWSER_TASK_NAMES = {
    "hand_landmarker.task": "hand_landmarker.task",
    "pose_landmarker.task": "pose_landmarker_lite.task",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Fetch to a temporary file first, so a failure cannot leave a partial asset."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as staged:
        staging_path = Path(staged.name)
    try:
        with urllib.request.urlopen(url, timeout=180) as response, staging_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        staging_path.replace(destination)
    except BaseException:
        staging_path.unlink(missing_ok=True)
        raise


def obtain(name: str, spec: dict, destination: Path, force: bool) -> bool:
    """Return True on success. A mismatched download is deleted, never kept."""
    if destination.is_file() and not force and sha256_file(destination) == spec["sha256"]:
        print(f"  {name}: already present and verified")
        return True

    print(f"  {name}: downloading ({spec['note']})")
    try:
        download(spec["url"], destination)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  {name}: download failed: {exc}", file=sys.stderr)
        return False

    actual = sha256_file(destination)
    if actual != spec["sha256"]:
        destination.unlink(missing_ok=True)
        print(
            f"  {name}: SHA-256 MISMATCH, file removed\n"
            f"      expected {spec['sha256']}\n"
            f"      actual   {actual}\n"
            f"      Upstream may have republished this asset. Do not use a mismatched\n"
            f"      file: inference would no longer match the training conditions.",
            file=sys.stderr,
        )
        return False
    print(f"  {name}: verified {actual[:16]}...")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--vendor-dir", type=Path, default=Path("webservice") / "vendor" / "mediapipe"
    )
    parser.add_argument("--python", action="store_true", help="only the Python .task models")
    parser.add_argument("--browser", action="store_true", help="only the browser web assets")
    parser.add_argument("--force", action="store_true", help="re-download even if verified")
    args = parser.parse_args(argv)

    want_python = args.python or not args.browser
    want_browser = args.browser or not args.python
    failures = 0

    if want_python:
        print(f"Python landmarkers -> {args.models_dir}")
        args.models_dir.mkdir(parents=True, exist_ok=True)
        for name, spec in PYTHON_MODELS.items():
            failures += not obtain(name, spec, args.models_dir / name, args.force)

    if want_browser:
        print(f"\nBrowser assets ({TASKS_VISION_VERSION}) -> {args.vendor_dir}")
        for name, spec in BROWSER_ASSETS.items():
            failures += not obtain(name, spec, args.vendor_dir / name, args.force)

        # The page loads the landmarkers from /vendor/mediapipe/, so copy rather
        # than re-download what models/ already holds.
        for source_name, vendor_name in BROWSER_TASK_NAMES.items():
            source = args.models_dir / source_name
            target = args.vendor_dir / vendor_name
            if not source.is_file():
                if not obtain(source_name, PYTHON_MODELS[source_name], source, args.force):
                    failures += 1
                    continue
            if not target.is_file() or sha256_file(target) != sha256_file(source):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                print(f"  {vendor_name}: copied from {args.models_dir}")
            else:
                print(f"  {vendor_name}: already present and verified")

    if failures:
        print(f"\n{failures} asset(s) unavailable; see models/README.md", file=sys.stderr)
        return 1
    print("\nall assets verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
