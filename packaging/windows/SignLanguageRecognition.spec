from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs


ROOT = Path(SPECPATH).parents[1]
mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all(
    "mediapipe"
)
torch_binaries = collect_dynamic_libs("torch")

# Both bundles ship. windows_entry.py still launches the legacy 27-class app,
# which resolves PreviewPaths.legacy_bundle_dir, so omitting the legacy bundle
# produces an executable that cannot find its own model.
datas = mediapipe_datas + [
    (str(ROOT / "models"), "resources/models"),
    (
        str(ROOT / "artifacts" / "realtime" / "best_current"),
        "resources/artifacts/realtime/best_current",
    ),
    (
        str(ROOT / "artifacts" / "legacy" / "daily30_27class"),
        "resources/artifacts/legacy/daily30_27class",
    ),
]

analysis = Analysis(
    [str(ROOT / "packaging" / "windows" / "windows_entry.py")],
    pathex=[str(ROOT)],
    binaries=mediapipe_binaries + torch_binaries,
    datas=datas,
    hiddenimports=mediapipe_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["paramiko"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SignLanguageRecognition",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SignLanguageRecognition",
)
