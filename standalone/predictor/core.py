"""Standalone U-Net ONNX predictor: GeoTIFF in -> prediction GeoTIFF out.

Self-contained (onnxruntime + rasterio + numpy only): no torch, no imports from
the app's `backend/`. The inference mirrors the in-app path exactly
(SegmentationModel.predict_image + SentinelImage normalization + tiling +
blend_tiles), so a prediction here matches what the app produces.

Label space, as everywhere in the project: 0 target, 1 background,
255 unlabeled/nodata.
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import rasterio

UNLABELED = 255
_DEFAULT_NODATA = -9999.0
_BATCH = 16


def default_model_path():
    """`model.onnx` next to the executable (frozen) or this package (source)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "model.onnx")


def load_model(onnx_path):
    """Open the ONNX session and read the model's embedded profile.

    Returns {session, in_channels, patch_size, band_names}. Band count and patch
    size come from the metadata written by the app's export_onnx; band_names
    falls back to generic names if absent/mismatched.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map or {}
    in_channels = int(meta.get("in_channels") or 0) or _input_channels(session)
    patch_size = int(meta.get("patch_size") or 96)
    try:
        band_names = json.loads(meta.get("band_names", "[]"))
    except (ValueError, TypeError):
        band_names = []
    if len(band_names) != in_channels:
        band_names = [f"Band {i + 1}" for i in range(in_channels)]
    return {"session": session, "in_channels": in_channels,
            "patch_size": patch_size, "band_names": band_names}


def _input_channels(session):
    shape = session.get_inputs()[0].shape  # [N, C, H, W]
    return int(shape[1]) if isinstance(shape[1], int) else 0


def read_bands_info(in_path):
    """Band count / descriptions / nodata of a GeoTIFF, for the band picker."""
    with rasterio.open(in_path) as src:
        return {"count": src.count,
                "descriptions": [d or "" for d in src.descriptions],
                "nodata": src.nodata,
                "size": [src.width, src.height]}


def _normalize(raw, nodata):
    """Per-band 2-98 percentile robust scaling to [0, 1]; nodata -> 0.

    Mirrors SentinelImage._percentiles/_normalize. `valid` is where every band
    is not nodata; percentiles are taken over valid pixels only.
    """
    valid = np.all(raw != nodata, axis=0)  # (H, W)
    bands = raw.shape[0]
    lo = np.zeros(bands, np.float32)
    hi = np.ones(bands, np.float32)
    for b in range(bands):
        vals = raw[b][valid]
        if vals.size:
            lo[b], hi[b] = np.percentile(vals, (2, 98))
            if hi[b] <= lo[b]:
                hi[b] = lo[b] + 1.0
    norm = np.clip((raw - lo[:, None, None]) / (hi - lo)[:, None, None], 0.0, 1.0)
    norm[:, ~valid] = 0.0
    return norm.astype(np.float32), valid


def _patch_grid(height, width, size):
    """Top-left corners of an overlapping size×size grid (stride size//2)."""
    stride = size // 2
    ys = list(range(0, max(height - size, 0) + 1, stride))
    xs = list(range(0, max(width - size, 0) + 1, stride))
    if ys[-1] != height - size:
        ys.append(height - size)
    if xs[-1] != width - size:
        xs.append(width - size)
    return [(y, x) for y in ys for x in xs]


def predict_geotiff(onnx_path, in_path, out_path, band_map=None, nodata=None,
                    write_probs=False, progress=None):
    """Predict a GeoTIFF and write a 1-band uint8 class-map GeoTIFF.

    band_map: 1-based GeoTIFF band indices in model-channel order (length must
    equal in_channels). Defaults to the first `in_channels` bands.
    nodata: overrides the source nodata (falls back to src.nodata, then -9999).
    write_probs: also write a float32 P(target) raster (`*_prob.tif`).
    progress: optional callback(done_tiles, total_tiles).
    Returns a summary dict.
    """
    model = load_model(onnx_path)
    session = model["session"]
    channels = model["in_channels"]
    size = model["patch_size"]

    with rasterio.open(in_path) as src:
        if band_map is None:
            if src.count < channels:
                raise ValueError(
                    f"Image has {src.count} band(s) but the model needs {channels}")
            band_map = list(range(1, channels + 1))
        if len(band_map) != channels:
            raise ValueError(
                f"Band mapping needs {channels} entries, got {len(band_map)}")
        if any(b < 1 or b > src.count for b in band_map):
            raise ValueError(f"Band indices must be within 1..{src.count}")
        nd = nodata if nodata is not None else (
            src.nodata if src.nodata is not None else _DEFAULT_NODATA)
        raw = src.read(band_map).astype(np.float32)  # (C, H, W)
        crs, transform = src.crs, src.transform

    _, height, width = raw.shape
    if min(height, width) < size:
        raise ValueError(
            f"Image is {width}x{height}, smaller than the {size}px patch size")

    norm, valid = _normalize(raw, nd)
    corners = _patch_grid(height, width, size)
    acc = np.zeros((1, height, width), np.float32)
    weight = np.zeros((height, width), np.float32)

    total = len(corners)
    done = 0
    for i in range(0, total, _BATCH):
        chunk = corners[i:i + _BATCH]
        batch = np.stack([norm[:, y:y + size, x:x + size] for y, x in chunk])
        logits = session.run(None, {"input": batch})[0]  # (b, 1, s, s)
        for (y, x), logit in zip(chunk, logits):
            acc[:, y:y + size, x:x + size] += logit
            weight[y:y + size, x:x + size] += 1.0
        done += len(chunk)
        if progress:
            progress(done, total)

    weight = np.maximum(weight, 1e-6)
    logit_map = acc / weight
    # Stable sigmoid; clip avoids overflow warnings at extreme logits (±60 is
    # already 0/1 to float precision, matching torch.sigmoid).
    p_target = 1.0 / (1.0 + np.exp(-np.clip(logit_map[0], -60.0, 60.0)))
    class_map = (p_target <= 0.5).astype(np.uint8)  # 0 target, 1 background
    class_map[~valid] = UNLABELED

    _write_geotiff(out_path, class_map, crs, transform, "uint8", UNLABELED)
    outputs = {"prediction": out_path}
    if write_probs:
        prob_path = os.path.splitext(out_path)[0] + "_prob.tif"
        prob = p_target.astype(np.float32)
        prob[~valid] = np.nan
        _write_geotiff(prob_path, prob, crs, transform, "float32", float("nan"))
        outputs["probability"] = prob_path

    return {"outputs": outputs, "tiles": total, "size": [width, height],
            "band_map": band_map, "target_pixels": int((class_map == 0).sum()),
            "valid_pixels": int(valid.sum())}


def _write_geotiff(path, arr, crs, transform, dtype, nodata):
    """1-band GeoTIFF preserving crs/transform (mirrors write_mask_geotiff)."""
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
