# Sentinel-2 Interactive Segmentation
![screenshot.png](readme_assets/screenshot.png)
## Disclaimer

This is a prototype for human-in-the-loop (binary) semantic segmentation in
remote sensing with semi-supervised training.
I build this after some exciting talks in a conference. **I also used
AI-Code generation in this project to get started.**

The tool turned out to be very useful to my work in remote sensing, and I
hope it is useful for others too.
**At the current state I can't guarantee that it works on every machine!**


Workflow: paint ground-truth strokes on the right pane, click
**Train**, and watch the live model inference refresh on the left pane.
Iterate until the prediction is good, then export the model (`.pth`/`.onnx`),
a standalone executable, or the merged mask.

## Setup & run

```bash
python3 -m venv .venv
# CPU-only torch (no CUDA download); drop the extra index on a GPU machine
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/pip install "pywebview[qt]"   # bundled Qt renderer, no system GTK needed
cd frontend && npm install && npm run build && cd ..
.venv/bin/python backend/main.py [path/to/image.tiff | path/to/project_dir]   # defaults to 00.tiff
```

The SAM2 click-assist tool needs `sam2/onnx/*.onnx` (regenerate with
`backend/export_sam_onnx.py`, see its docstring); without them the SAM2
button stays disabled. The standalone predictor and "Export Executable" need
`pyinstaller` and a build of `standalone/predictor.spec` — see
`standalone/README.md`.

## Structure

```
backend/
  main.py            entry point
  api.py              JS bridge
  model.py            U-Net, pre-trained 
  train.py            semi-supervised loop 
  data.py              GeoTIFF I/O, normalization, patch handling
  project.py           config, tiling, autosave
  report.py            accuracy report generation
  sam_service.py       SAM2 click-to-segment assist (ONNX runtime)
  export_sam_onnx.py   exports Efficient-SAM2 
  
frontend/src/          React (Vite)
  App.jsx               top-level state
  App.css               all app styling
  bridge.js             thin wrapper around window.pywebview.api
  constants.js           classes, colors, unlabeled value
  main.jsx               React root
  
  components/
    Viewport.jsx          shared pan/zoom canvas used by both panes
    InferencePane.jsx      left pane
    LabelerPane.jsx        right pane
    Toolbar.jsx             top toolbar: menus, tools, brush size etc.
    Dropdown.jsx            toolbar dropdown menu
    ContextMenu.jsx         mouse-anchored context menu 
    ThumbnailStrip.jsx      project image thumbnails 
    ProjectSetup.jsx        new/edit project dialog 
    AccuracyReport.jsx      validation vs. training IoU report
    
standalone/             self-contained ONNX predictor (GUI + CLI)
                        
sam2/onnx/              exported SAM2 encoder/decoder (*.onnx), required for the SAM2 tool

pretraining/pretrained.pth  default weights the model resets to

project-template.json  starter config for new multi-image projects

00.tiff                 sample 10-band stack (512x512, EPSG:32635)
```

## How it works
### 1. Data & Classes

- Target (0): Orange.
- Background (1): Blue.
- Unlabeled (255): Ignored during training; treated as "nodata."
- Preprocessing: Per-band 2nd–98th percentile scaling to [0, 1] range, ignoring nodata.

### 2. Model Architecture

- Core: U-Net using a timm-efficientnet-b0 encoder.
- Input: Raw bands, processed in 96×96 patches (default).
- Inference: Uses overlapping tiles with logit averaging for smooth, full-image results.
- Sharpening: Optional PointRend head for cleaner boundaries on uncertain pixels.

### 3. Training Strategy

- Supervised: Standard cross-entropy on user-labeled pixels (user input overrides model).
- Semi-supervised: FixMatch-style pseudo-labeling on confident (>0.9) predictions.
- Consistency: MSE between weak/strong augmentations; loss terms ramp up over 5 epochs.

### 4. SAM2 Assist

- Efficient-SAM2 exported to ONNX.
- Runs via onnxruntime: no GPU or heavy PyTorch dependencies required for click-to-segment.

### 5. Exports

- Model/State Dict: Saves training weights.
- ONNX/Executable: Standalone predictor reproducing app inference (tiling, scaling, thresholding).
- Mask: 1-band uint8 GeoTIFF (prediction + user labels) with original CRS/affine transform.## Frontend development

```bash
cd frontend && npm run dev   # hot-reload UI (bridge calls need the pywebview host)
npm run build                # rebuild dist/ used by backend/main.py
```