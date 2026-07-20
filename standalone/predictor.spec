# PyInstaller spec for the standalone U-Net predictor (onedir).
#
# Build:  cd standalone && ../venv/bin/python -m PyInstaller predictor.spec --noconfirm
# Result: standalone/dist/predictor/  (predictor executable + _internal/).
# The executable is model-agnostic: drop a `model.onnx` next to it at runtime.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# rasterio bundles its own GDAL + data (gdal_data/proj_data); onnxruntime ships
# native libs. collect_all pulls their data files, binaries and submodules.
for pkg in ("rasterio", "onnxruntime"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += ["predictor", "predictor.core", "predictor.cli", "predictor.gui"]

a = Analysis(
    ["entry.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # The predictor never imports these; exclude them so a stray transitive
    # reference can't pull hundreds of MB of unused ML stack into the bundle.
    excludes=["torch", "torchvision", "matplotlib", "scipy", "pandas",
              "IPython", "backend", "webview", "segmentation_models_pytorch"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="predictor",
    console=True,  # keep stdout for the CLI; the GUI still opens its own window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="predictor",
)
