# PyInstaller spec for the full Kaio-ken Segmenter desktop app (onedir).
#
# Build:  ./build_app.sh   (or: venv/bin/python -m PyInstaller app.spec --noconfirm)
# Result: dist/kaioken-segmenter/  (executable + _internal/).
#
# Bundles backend/ (pywebview host + torch training/inference) together with
# the built frontend/dist/ (React UI). Run `cd frontend && npm run build`
# first -- build_app.sh does this automatically if dist/ is missing.
#
# torch, torchvision, cv2, onnxruntime and pywebview all ship their own
# PyInstaller hooks (via pyinstaller-hooks-contrib / their own package), so
# only rasterio -- which has none -- needs a manual collect_all.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("rasterio")

datas += [("frontend/dist", "frontend/dist")]

# backend/*.py import each other as top-level modules (main.py adds
# backend/ to sys.path at runtime); pathex below lets Analysis find them,
# but list them explicitly too since they're imported dynamically by api.py.
hiddenimports += ["api", "data", "model", "project", "report", "sam_service", "train"]

a = Analysis(
    ["backend/main.py"],
    pathex=["backend"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # export_sam_onnx.py is a standalone dev script (SAM2 -> ONNX export),
    # never imported by the running app; keep its extra deps out of the bundle.
    excludes=["export_sam_onnx"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="kaioken-segmenter",
    console=True,  # keep stdout for logs/tracebacks
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="kaioken-segmenter",
)
