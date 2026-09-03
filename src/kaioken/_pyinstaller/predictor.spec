# PyInstaller spec for the standalone U-Net predictor (onedir).
#
# Not run by hand: the app's "Export Executable" invokes it (see
# api.export_executable), passing its own --distpath/--workpath. Requires the
# export extra:  pip install "kaioken-segmenter[export]"
#
# Result: <distpath>/predictor/  (predictor executable + _internal/).
# The executable is model-agnostic: drop a `model.onnx` next to it at runtime.

import os

from PyInstaller.utils.hooks import collect_all

# SPECPATH is <kaioken>/_pyinstaller, so its grandparent is the directory holding
# the kaioken package -- what `import kaioken.predictor` needs on the path.
PACKAGE_DIR = os.path.dirname(SPECPATH)
IMPORT_ROOT = os.path.dirname(PACKAGE_DIR)
ENTRY = os.path.join(PACKAGE_DIR, "predictor", "_entry.py")

datas, binaries, hiddenimports = [], [], []
# rasterio bundles its own GDAL + data (gdal_data/proj_data); onnxruntime ships
# native libs. collect_all pulls their data files, binaries and submodules.
for pkg in ("rasterio", "onnxruntime"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += ["kaioken.predictor", "kaioken.predictor.core",
                  "kaioken.predictor.cli", "kaioken.predictor.gui"]

a = Analysis(
    [ENTRY],
    pathex=[IMPORT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # The predictor never imports these; exclude them so a stray transitive
    # reference can't pull hundreds of MB of unused ML stack into the bundle.
    # The app's own modules are listed for the same reason -- they live in the
    # same package as the predictor, so they are one careless import away.
    excludes=["torch", "torchvision", "matplotlib", "scipy", "pandas",
              "IPython", "webview", "segmentation_models_pytorch",
              "kaioken.api", "kaioken.app", "kaioken.cli", "kaioken.assets",
              "kaioken.data", "kaioken.model", "kaioken.project",
              "kaioken.report", "kaioken.sam_service", "kaioken.train",
              "kaioken.wand"],
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
