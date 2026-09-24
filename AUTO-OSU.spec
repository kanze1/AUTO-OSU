# PyInstaller spec for the Windows build.  Usage:  scripts/build_exe.ps1  (or  pyinstaller AUTO-OSU.spec)
#
# One-folder build: dist/AUTO-OSU/AUTO-OSU.exe plus its libraries.  The trained models are NOT
# bundled here; the release zip adds a models/ folder next to the exe (see scripts/build_exe.ps1),
# and the app can also download them on first run.
import os
import sys
from pathlib import Path
from scripts.prepare_runtime_bundle import write_bundle
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

block_cipher = None
datas, binaries, hiddenimports = [], [], []
for pkg in ("customtkinter", "tkinterdnd2", "librosa", "imageio_ffmpeg", "soundfile"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("autoosu")
if sys.version_info >= (3, 11):
    datas += copy_metadata("rosu-pp-py")
datas += [("autoosu/assets", "autoosu/assets")]
runtime_bundle = write_bundle(Path(os.getcwd()), Path(os.getcwd()) / "build/runtime-source.zip")
datas += [(str(runtime_bundle), "runtime")]
hiddenimports += ["scipy.special._cdflib", "scipy._lib.array_api_compat.numpy.fft", "sklearn.utils._typedefs"]

a = Analysis(
    ["launcher.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "wandb", "IPython", "jupyter", "notebook", "pytest", "torchvision", "torchaudio",
              "triton", "PyQt5", "PySide6", "tensorboard", "sklearn"],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AUTO-OSU",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icon.ico" if os.path.exists("assets/icon.ico") else None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="AUTO-OSU")
