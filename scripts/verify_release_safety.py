from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


BANNED_MARKERS = [
    b"163.13." + b"202.125",
    b"tku" + b"310ai",
    b"b310ai",
    b"C:\\Users\\User",
    b"remote_run_dir",
    b"SLR_REMOTE_" + b"PASSWORD",
    b"paramiko",
]


def find_markers_in_stream(stream, chunk_size: int = 1024 * 1024) -> set[bytes]:
    found: set[bytes] = set()
    overlap = max(len(marker) for marker in BANNED_MARKERS) - 1
    previous = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        haystack = (previous + chunk).lower()
        for marker in BANNED_MARKERS:
            if marker.lower() in haystack:
                found.add(marker)
        previous = haystack[-overlap:]
    return found


def scan_portable_directory(path: Path) -> list[str]:
    failures: list[str] = []
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        lowered_name = file_path.as_posix().lower()
        if "paramiko" in lowered_name or "/ssh" in lowered_name:
            failures.append(f"SSH-related filename: {file_path}")
        with file_path.open("rb") as stream:
            for marker in find_markers_in_stream(stream):
                failures.append(
                    f"Marker {marker.decode('ascii', errors='replace')!r} in {file_path}"
                )
    return failures


def scan_zip(path: Path) -> list[str]:
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            lowered_name = member.filename.lower()
            if "paramiko" in lowered_name or "/ssh" in lowered_name:
                failures.append(f"SSH-related ZIP member: {member.filename}")
            if member.is_dir():
                continue
            with archive.open(member) as stream:
                for marker in find_markers_in_stream(stream):
                    failures.append(
                        f"Marker {marker.decode('ascii', errors='replace')!r} in ZIP member {member.filename}"
                    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if a portable release contains credentials, personal paths, or SSH code."
    )
    parser.add_argument("--portable-dir", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    args = parser.parse_args()

    failures = scan_portable_directory(args.portable_dir) + scan_zip(args.zip_path)
    if failures:
        raise SystemExit("\n".join(failures))
    print("Release safety scan passed.")


if __name__ == "__main__":
    main()
