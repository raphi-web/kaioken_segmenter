# Sentinel-2 Interactive Segmentation

Desktop prototype for human-in-the-loop semantic segmentation of 10-band
Sentinel-2 imagery (Red, Green, Blue, NIR1, RE1, RE2, RE3, NIR2, SWIR1, SWIR2)
with semi-supervised training.

Two-pane workflow: paint ground-truth strokes on the right pane, click
**Train**, and watch the live model inference refresh on the left pane.
Iterate until the prediction is good, then export the model (`.pth`) or the
merged mask (GeoTIFF with CRS/affine metadata).

## Structure

```
backend/
  main.py     entry point: pywebview window hosting the built frontend
  api.py      JS bridge (image/overlay transfer, labels, training, exports)
  model.py    U-Net, pre-trained EfficientNet-B0 encoder, 10 input channels
  train.py    semi-supervised loop (CE + pseudo-labels + consistency)
  data.py     GeoTIFF I/O, robust normalization, 96x96 patch handling
frontend/     React (Vite): InferencePane, LabelerPane, Toolbar
00.tiff       sample 10-band stack (512x512, EPSG:32635)
```

## Setup & run

```bash
python3 -m venv .venv
# CPU-only torch (no CUDA download); drop the extra index on a GPU machine
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/pip install "pywebview[qt]"   # bundled Qt renderer, no system GTK needed
cd frontend && npm install && npm run build && cd ..
.venv/bin/python backend/main.py [path/to/image.tiff]   # defaults to 00.tiff
```

## Controls

- **Zoom:** mouse wheel over either pane, centered on the cursor. Zoom and
  pan are synced between the two panes.
- **Pan:** middle-button drag on either pane; left-drag also pans on the
  inference (left) pane. On the labeler, left-drag paints.
- **Reset view:** double-click the inference pane to re-fit the image.
- **Tools:** Brush paints strokes; Polygon fills clicked outlines (click to add
  points, Enter/double-click to fill, Esc to cancel). The Eraser applies to
  both. Undo (button or Ctrl+Z) reverts the last 50 strokes/fills.
- **SAM2:** click an object to select it, then refine before committing —
  left-click adds an *include* point (pink dot), right-click (or Ctrl/Alt+left)
  adds an *exclude* point (red dot with a bar) that shrinks the selection away
  from that spot. Backspace drops the last point, Enter fills the selection
  with the current class, Esc discards it. Nothing is painted until Enter, so
  the mask can be narrowed as much as needed first.
- **Uncertainty:** checkbox on the inference pane overlays a yellow heatmap of
  1 − max class probability — label there first.
- **Adopt Prediction:** fills every unlabeled pixel with the current model
  prediction so you can correct it by hand; painted pixels are never
  overwritten and the whole step is one Undo entry.
- **Clear Labels:** removes all labels from the current image (one Undo
  entry as well).
- **From GeoTIFF…:** builds a project out of a single large raster. Pick the
  file, then set tile width/height and overlap (in px); the raster is cut into
  georeferenced tiles in the output folder, which becomes the project. Each
  tile keeps the source CRS, dtype, band count and nodata with its own affine
  transform. The last row/column is snapped back to the edge so every tile is
  full-size, tiles must be at least the patch size, and all-nodata tiles are
  skipped. The output folder must be new or empty.
- **Projects:** New / Open picks a folder of GeoTIFFs; folders without a
  `project_config.json` get a setup dialog (bands, band names, patch size,
  folders). Everything persists in the project: labels autosave to
  `masks_user/` on image switch, and the trained model + epoch counter
  autosave to `model.pt` in the project root after every training run and are
  restored when the project is reopened.
- **Reset Model:** discards all training (including the project's `model.pt`)
  and returns to the pretrained default weights (`pretraining/pretrained.pth`,
  produced by `pretraining/pretrain.py`).

## How it works

- **Classes:** 0 target (orange), 1 background (blue, the anti-target:
  everything that is not target) — the same values in the UI, the masks on
  disk and the model's output channels. Unpainted pixels are "unlabeled"
  (255, also the mask GeoTIFFs' nodata tag) and are ignored by the supervised
  loss; source-nodata pixels are excluded from every loss term.
- **Model:** a `segmentation_models_pytorch` U-Net with a
  `timm-efficientnet-b0` ImageNet encoder directly on the raw input bands;
  patches sized by the project's data profile (default 96×96), overlapping
  tiles with logit averaging for full-image inference.
- **Normalization:** per-band 2nd–98th percentile scaling to [0, 1] over valid
  (non-nodata) pixels, cached per image.
- **Semi-supervised loss** (FixMatch-style, Albumentations augs):
  supervised cross-entropy on user pixels; pseudo-labeling on confident
  (>0.9) unlabeled pixels; consistency (MSE) between weak/strong views.
  Unsupervised terms ramp up over the first 5 cumulative epochs.
  User-labeled pixels always override model beliefs — in training and in the
  exported mask.
- **Exports:** "Export Model" saves the `state_dict`; "Export Mask" writes a
  1-band uint8 GeoTIFF (prediction merged with user labels) carrying the
  source CRS and affine transform.

## Frontend development

```bash
cd frontend && npm run dev   # hot-reload UI (bridge calls need the pywebview host)
npm run build                # rebuild dist/ used by backend/main.py
```
