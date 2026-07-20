# Standalone U-Net predictor

A self-contained predictor that runs the exported U-Net (`model.onnx`) on a
GeoTIFF — no Python needed. Depends only on **onnxruntime + rasterio + numpy**
(and stdlib tkinter for the GUI); it does **not** import the app's `backend/`.

Inference matches the in-app path exactly (normalization, overlapping-tile
blending, thresholding), verified at 100% pixel agreement with
`SegmentationModel.predict_image`.

## Use from source

```bash
# GUI (pick image, map bands, run):
../venv/bin/python -m predictor

# CLI:
../venv/bin/python -m predictor scene.tif -m model.onnx \
    --bands 1,2,3,4,5,6,7,8,9,10 -o scene_prediction.tif [--probs]
```

Model resolution: `--model` if given, else `model.onnx` next to the executable
(or the `standalone/` dir when run from source). Band count / patch size /
band names are read from the ONNX metadata written by the app's `Export ONNX`.
Output is a 1-band uint8 GeoTIFF (0 target, 1 background, 255 nodata) with the
input's CRS/transform preserved.

## Build the clickable executable

```bash
cd standalone
../venv/bin/python -m pip install pyinstaller   # once (build-only)
../venv/bin/python -m PyInstaller predictor.spec --noconfirm
# -> dist/predictor/  (predictor executable + _internal/)
```

The binary is **model-agnostic** (loads a sibling `model.onnx` at runtime) and
**OS-specific** (built here as a Linux ELF). Once built, the app's
**Export / Model → Export Executable** copies `dist/predictor/` into a folder
the user picks and drops the current model as `model.onnx` beside the binary;
`Api.executable_available()` disables that button until the build exists.

Double-click behavior: no arguments → GUI; arguments → CLI (`entry.py` dispatch).
