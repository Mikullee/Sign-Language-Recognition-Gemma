"""Verify an extracted Knee42 Mac auto-trigger handoff package."""
from pathlib import Path

try:
    from scripts.build_mac_auto_trigger_package import validate_release_tree
except ModuleNotFoundError:  # Direct invocation from an extracted package.
    from build_mac_auto_trigger_package import validate_release_tree


if __name__ == "__main__":
    validate_release_tree(Path(__file__).resolve().parents[1])
    print("Mac auto-trigger package verified: 42-class model, runtime assets, and videos are present.")
