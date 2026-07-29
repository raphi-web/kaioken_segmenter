"""pywebview bridge: the Api class is exposed to JS as window.pywebview.api.

Every method is called by pywebview on a worker thread and returns
JSON-serializable data, so each JS call resolves as a promise.
"""

import base64
import os
import sys
import threading
import time
import traceback

import numpy as np
import rasterio
import torch
import webview

try:
    import cv2  # only for the wand's connected-component pass; see _connected_region
except ImportError:  # pragma: no cover - opencv is a project dependency
    cv2 = None

import report as report_module
import train as train_module
from data import (UNLABELED, SentinelImage, array_to_png_b64, tile_geotiff,
                  write_mask_geotiff)
from model import SegmentationModel
from project import (DEFAULT_CLASSES, SPLIT_ROLES, Project,
                     config_from_settings, config_path, default_config,
                     default_split, deterministic_validation,
                     probe_band_count, save_config)
from sam_service import (FEATURE_LEVELS, SamService, mask_to_polygons,
                         resize_bilinear)

# Alpha of the prediction overlay (unlabeled/no-data stays transparent); the
# RGB part comes from the project's (or standalone image's) class palette.
OVERLAY_ALPHA = 160


# ---------- magic wand ----------

# Offsets of a 5x5 window ordered nearest-first, so _sample_colour can take the
# N pixels closest to a click by slicing the front of this list. The click
# itself is offset 0, hence always included.
_SAMPLE_OFFSETS = sorted(
    ((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)),
    key=lambda o: (o[0] * o[0] + o[1] * o[1], o[0], o[1]),
)
MAX_SAMPLE_PIXELS = len(_SAMPLE_OFFSETS)  # 25


def _hex_to_rgba(hex_color, alpha=OVERLAY_ALPHA):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


# Prebuilt standalone predictor (PyInstaller onedir). "Export Executable" copies
# this folder and drops the current model.onnx beside the binary.
#
# Source tree: standalone/dist/predictor (from standalone/build.sh).
# Frozen app: standalone/predictor next to this executable -- build_app.sh
# copies standalone/dist/predictor there so both builds serve the same feature.
if getattr(sys, "frozen", False):
    _PREDICTOR_DIR = os.path.join(
        os.path.dirname(sys.executable), "standalone", "predictor")
else:
    _PREDICTOR_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "standalone", "dist", "predictor")


class Api:
    def __init__(self, image_path=None, project_root=None):
        self._project = None
        self._image = None
        self._image_name = None
        self._labels = None
        self._model = None  # built to match the data profile below
        self._class_map = None  # cached last full-image prediction
        self._probs = None      # matching (C, H, W) softmax probabilities
        # Bumped whenever the model's weights (or the open project) change, so
        # generate_accuracy_report can tell a cached report is still valid
        # instead of rescoring every image again for nothing.
        self._model_version = 0
        self._dirty_labels = {}  # name -> label buffer, edited this session and not yet saved
        self._dirty_geo = {}     # name -> (crs, transform) for each dirty entry above
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sam = SamService()  # Efficient-SAM2 click assist, independent of _model
        # Magic wand caches, both keyed by image name so a stale field can never
        # outlive the image it describes (see _clear_wand_cache).
        # _wand_cache:      (source, level, y, x, sample_size) -> full-image field
        # _sam_stack_cache: level -> centred unit-length features, so each new
        #                   sample is one dot product rather than a re-prepare.
        self._wand_cache = (None, {})
        self._sam_stack_cache = (None, {})
        # Background GeoTIFF tiling (see create_project_from_geotiff)
        self._tiling = {"running": False, "done": 0, "total": 0,
                        "result": None, "error": None}
        # Background accuracy report (see generate_accuracy_report). Same shape
        # as _tiling; result holds the last finished report so save_accuracy_report
        # writes exactly what the modal is showing.
        self._report = {"running": False, "done": 0, "total": 0,
                        "result": None, "error": None, "model_version": None}
        self._report_stop = threading.Event()
        self._status = {"state": "idle", "epoch": 0, "epochs": 0, "total_epochs": 0,
                       "images": 0, "ramp": None, "stage": None,
                       "scan_done": 0, "scan_total": 0,
                       "loss_sup": None, "loss_pseudo": None, "loss_cons": None, "error": None}
        if project_root:
            result = self._init_project(project_root)
            if not result["ok"]:
                raise ValueError(result["error"])
        elif image_path:
            # Standalone image (no project): profile the model on the image itself.
            image = SentinelImage(image_path)
            self._model = SegmentationModel(in_channels=image.bands)
            self._set_image(image, os.path.basename(image_path))
        if self._model is None:
            self._model = SegmentationModel()

    def _set_image(self, image, name, labels=None):
        """Swap the active image and its label buffer; invalidate prediction caches."""
        self._image = image
        self._image_name = name
        if image is not None and labels is None:
            labels = np.full((image.height, image.width), UNLABELED, dtype=np.uint8)
        self._labels = labels
        self._class_map = None
        self._probs = None
        self._clear_wand_cache()

    def _clear_wand_cache(self):
        """Drop every cached wand field. Called whenever the pixels change.

        Both caches describe one specific image; keeping them across a switch
        would let a click on the new image be answered from the old one's
        distances, which is silently wrong rather than visibly broken.
        """
        self._wand_cache = (self._image_name, {})
        self._sam_stack_cache = (self._image_name, {})

    def _display_bands(self):
        """The project's configured [R, G, B] display band indices, or None
        (which makes rgb_composite fall back to the first three bands)."""
        if self._project is not None:
            return self._project.config["data_profile"].get("display_bands")
        return None

    def _image_payload(self):
        """Everything the frontend needs to display the active image, or None."""
        if self._image is None:
            return None
        return {
            "name": self._image_name,
            "png": array_to_png_b64(self._image.rgb_composite(self._display_bands())),
            "width": self._image.width,
            "height": self._image.height,
            "labels": base64.b64encode(self._labels.tobytes()).decode("ascii"),
            "labeled_pixels": int((self._labels != UNLABELED).sum()),
        }

    # ---------- image / overlay ----------

    def _overlay_colors(self):
        """id -> RGBA for the prediction overlay: the open project's class
        palette, or the standalone-image default when no project is open."""
        classes = self._project.config["classes"] if self._project else DEFAULT_CLASSES
        return {c["id"]: _hex_to_rgba(c["color"]) for c in classes}

    def get_image(self):
        return self._image_payload()

    def get_overlay(self):
        """Colorized RGBA PNG of the current model prediction (no-data transparent)."""
        if self._image is None:
            return {"png": None, "iou": None}
        if self._class_map is None:
            with self._lock:
                self._class_map, self._probs = self._model.predict_image(self._image)
        rgba = np.zeros((*self._class_map.shape, 4), dtype=np.uint8)
        for cls, color in self._overlay_colors().items():
            rgba[self._class_map == cls] = color
        return {"png": array_to_png_b64(rgba), "iou": self._target_iou()}

    def get_uncertainty(self):
        """Yellow RGBA heatmap PNG; alpha scales with 1 - max class probability."""
        if self._image is None:
            return {"png": None}
        if self._probs is None:
            with self._lock:
                self._class_map, self._probs = self._model.predict_image(self._image)
        uncertainty = 1.0 - self._probs.max(axis=0)  # in [0, 0.5] for two classes
        rgba = np.zeros((*uncertainty.shape, 4), dtype=np.uint8)
        rgba[..., :3] = (255, 220, 0)
        rgba[..., 3] = np.clip(uncertainty * 2.0 * 255.0, 0, 255).astype(np.uint8)
        rgba[~self._image.valid_mask] = 0
        return {"png": array_to_png_b64(rgba)}

    # ---------- labels ----------

    def _target_iou(self):
        """IoU of the target class (0) between the current model prediction and
        the user's labels, measured over the pixels the user has drawn.

        Returns None when the user has not drawn any target pixels or no
        prediction has been computed yet (uses the cached class map, never
        triggers a forward pass). Over the drawn region, pixels drawn as
        background but predicted target lower the score (false positives), and
        target pixels predicted background do too (false negatives).
        """
        if self._image is None or self._class_map is None:
            return None
        drawn = (self._labels != UNLABELED) & self._image.valid_mask
        gt_target = drawn & (self._labels == 0)
        if not gt_target.any():
            return None  # no target ground truth to score against
        pred_target = drawn & (self._class_map == 0)
        return round(float((gt_target & pred_target).sum() / (gt_target | pred_target).sum()), 4)

    def set_labels(self, b64):
        """Receive the raw label buffer (H*W uint8: 0 target, 1 background, 255 unlabeled)."""
        if self._image is None:
            return {"labeled_pixels": 0, "iou": None}
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        self._labels = buf.reshape(self._image.height, self._image.width).copy()
        self._dirty_labels[self._image_name] = self._labels
        self._dirty_geo[self._image_name] = (self._image.crs, self._image.transform)
        return {"labeled_pixels": int((self._labels != UNLABELED).sum()),
                "iou": self._target_iou()}

    def transfer_prediction(self):
        """Copy the model prediction into the user labels (painted pixels win).

        Only unlabeled valid pixels are filled, so existing ground truth is
        never overwritten; returns the new label buffer for the frontend.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        with self._lock:
            if self._class_map is None:
                self._class_map, self._probs = self._model.predict_image(self._image)
            take = (self._labels == UNLABELED) & (self._class_map != UNLABELED)
            self._labels[take] = self._class_map[take]
            self._dirty_labels[self._image_name] = self._labels
            self._dirty_geo[self._image_name] = (self._image.crs, self._image.transform)
        return {"ok": True,
                "labels": base64.b64encode(self._labels.tobytes()).decode("ascii"),
                "labeled_pixels": int((self._labels != UNLABELED).sum()),
                "iou": self._target_iou()}

    # ---------- SAM2 snap-to-edge assist ----------

    def sam_available(self):
        """Whether the Efficient-SAM2 click assist can run (files present)."""
        reason = self._sam.unavailable_reason()
        return {"available": reason is None, "reason": reason}

    def _sam_image_key(self):
        """Cache key for the SAM embeddings: anything that changes the RGB pixels."""
        root = self._project.root if self._project is not None else None
        bands = self._display_bands()
        return (root, self._image_name, tuple(bands) if bands else ())

    def sam_snap(self, points):
        """Segment from a list of clicks and return editable polygon vertices.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        reason = self._sam.unavailable_reason()
        if reason is not None:
            return {"ok": False, "error": reason}
        clicks = [(float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 1)
                  for p in (points or [])]
        if not clicks:
            return {"ok": False, "error": "No points given"}
        if all(lab == 0 for *_, lab in clicks):
            # SAM2 has nothing to grow from; the caller should keep the previous
            # mask rather than showing an empty one.
            return {"ok": False, "error": "Add an include point first"}
        try:
            rgb = self._image.rgb_composite(self._display_bands())  # (H, W, 3) uint8
            self._sam.set_image(rgb, self._sam_image_key())
            mask, score = self._sam.predict_points(clicks)
            polygons = mask_to_polygons(mask)
        except Exception as e:  # model/runtime errors must not crash the bridge
            traceback.print_exc()
            return {"ok": False, "error": f"SAM2 failed: {e}"}
        if not polygons:
            return {"ok": False, "error": "SAM2 found no object at those points"}
        return {"ok": True, "polygons": polygons, "score": score}

    # ---------- magic wand ----------
    #
    # Region-growing selection from clicks. The whole design serves one
    # property: ADDING A POSITIVE CLICK NEVER REMOVES A PIXEL. Each sample is
    # resolved completely independently -- its own reference, its own match set,
    # its own region, its own budget -- and the results are unioned, so nothing
    # about one sample can reach another. Negative clicks are the deliberate
    # exception; removing pixels is what they are for.

    def _sample_colour(self, y, x, sample_size):
        """Mean spectrum of the `sample_size` valid pixels nearest (y, x).

        Averaging denoises the reference: a one-pixel sample is a sample of size
        one, and on textured ground the clicked pixel is easily an outlier its
        own region then fails to match. Only valid pixels contribute, so a click
        beside a nodata edge averages real ground rather than the fill value.
        The clicked pixel is offset 0 of _SAMPLE_OFFSETS, so it always counts.

        Note this averages *within* one click only. Averaging across clicks
        would move the reference for everything, letting a new click drop pixels
        an earlier one had selected -- exactly the property the wand promises.
        """
        image = self._image
        stack = image.normalized
        picked = []
        for dy, dx in _SAMPLE_OFFSETS[:sample_size]:
            py, px = y + dy, x + dx
            if 0 <= py < image.height and 0 <= px < image.width and image.valid_mask[py, px]:
                picked.append(stack[:, py, px])
        if not picked:  # the click itself is nodata: fall back to its own value
            picked = [stack[:, y, x]]
        return np.mean(picked, axis=0)

    def _spectral_field(self, y, x, sample_size):
        """Per-pixel RMS band difference from the click's reference spectrum.

        Computed over EVERY band of the normalized stack, not the three-band RGB
        composite the pane displays: water and terrain shadow are near-identical
        in true colour and separate cleanly in NIR/SWIR, and the composite has
        discarded those bands before the frontend sees a pixel.

        RMS rather than a plain Euclidean norm so one tolerance means the same
        thing whether a project has 4 bands or 12.
        """
        ref = self._sample_colour(y, x, sample_size)[:, None, None]
        diff = self._image.normalized - ref
        return np.sqrt(np.mean(diff * diff, axis=0))

    def _sam_features(self, level):
        """Centred, unit-length SAM2 features for `level` as (C, h, w).

        CENTRING IS LOAD-BEARING, not hygiene. Raw SAM2 features share a large
        common component, so every pixel is far from every other: measured on a
        512px tile, click-to-scene distances ran 0.4-0.8 with the whole useful
        range inside the last tenth of that. Subtracting the image's own mean
        feature puts the field at 0 at the click and spreads it to ~1.5, giving
        the tolerance somewhere to work. The consequence, which the UI has to
        live with, is that a tolerance is relative to the variety in the tile in
        front of you rather than to SAM2's feature space at large.

        Normalizing here makes the cosine distance downstream one dot product.
        """
        name, cache = self._sam_stack_cache
        if name != self._image_name:  # defensive: _set_image should have cleared it
            cache = {}
            self._sam_stack_cache = (self._image_name, cache)
        if level not in cache:
            feats = np.asarray(self._sam.features(level), dtype=np.float32)
            feats = feats - feats.mean(axis=(1, 2), keepdims=True)
            norm = np.linalg.norm(feats, axis=0, keepdims=True)
            cache[level] = feats / np.maximum(norm, 1e-8)
        return cache[level]

    def _embedding_field(self, y, x, sample_size, level):
        """Per-pixel cosine distance from the click, in SAM2 feature space.

        COSINE, not Euclidean: feature magnitude carries no meaning here and
        varies wildly between the three levels; the angle does. (Same reasoning
        as the spectral field's RMS.)

        THE FIELD IS UPSAMPLED, NOT THE FEATURES. The distance is computed on
        the encoder's own grid and only the finished one-channel field is
        resized to image resolution -- upsampling `deep`'s 256 channels to a
        512px image would cost 268 MB where its field costs 1 MB, and nothing
        downstream can tell, since the flood fill, the valid mask and protection
        all still run at pixel resolution.

        A consequence worth knowing: after that upsample the exact zero at the
        reference cell survives nowhere (0.09 at `fine` to 0.44 at `deep`, at
        the clicked pixel itself). The wand does not depend on it -- the sample
        is seeded into its own match set regardless.
        """
        feats = self._sam_features(level)
        _, gh, gw = feats.shape
        # The encoder resizes to a square regardless of aspect, so a pixel maps
        # onto the grid by plain proportional scaling.
        gy = min(gh - 1, max(0, int(y / self._image.height * gh)))
        gx = min(gw - 1, max(0, int(x / self._image.width * gw)))
        picked = []
        for dy, dx in _SAMPLE_OFFSETS[:sample_size]:
            cy, cx = gy + dy, gx + dx
            if 0 <= cy < gh and 0 <= cx < gw:
                picked.append(feats[:, cy, cx])
        ref = np.mean(picked, axis=0)
        ref = ref / max(float(np.linalg.norm(ref)), 1e-8)
        field = 1.0 - np.tensordot(ref, feats, axes=(0, 0))  # (gh, gw)
        return resize_bilinear(field, (self._image.height, self._image.width))

    def _wand_fields(self, samples, sample_size, source, level):
        """[((y, x), field), ...] -- one full-image distance field per sample.

        `source` picks how distance is measured and nothing else in the wand
        changes with it: connectivity, the budget, the negatives and protection
        all consume only the field.

        Cached per (source, level, point, sample_size), so adding a click costs
        one new field, dragging the tolerance slider costs none, and flipping
        between the two sources costs nothing after the first look at each.
        """
        name, cache = self._wand_cache
        if name != self._image_name:
            cache = {}
            self._wand_cache = (self._image_name, cache)
        out = []
        for y, x in samples:
            key = (source, level, y, x, sample_size)
            if key not in cache:
                cache[key] = (self._embedding_field(y, x, sample_size, level)
                              if source == "sam"
                              else self._spectral_field(y, x, sample_size))
            out.append(((y, x), cache[key]))
        return out

    @staticmethod
    def _connected_region(within, seed):
        """The 4-connected component of `within` that contains `seed`.

        4-CONNECTED, NOT 8: with 8 the fill crosses diagonals, so two regions
        touching at a single corner pixel count as one and the selection leaks
        through gaps too narrow to see at the zoom you clicked at.

        Uses OpenCV's C implementation, which settles the common case on its
        own; the pure-Python breadth-first fallback (unbounded _grow) is only
        there so a missing cv2 degrades speed rather than the feature.
        """
        if cv2 is None:
            return Api._grow_within_budget(within, seed, within.size)
        count, labels = cv2.connectedComponents(
            within.astype(np.uint8), connectivity=4)
        return labels == labels[seed]

    @staticmethod
    def _grow_within_budget(within, seed, budget):
        """The `budget` pixels of `within` nearest `seed` THROUGH the region.

        Breadth-first from the seed, so pixels arrive in order of path distance
        and the result is the innermost N and always connected. Taking the
        nearest N by straight-line distance instead would return two halves of a
        horseshoe with the middle missing.
        """
        height, width = within.shape
        sy, sx = seed
        out = np.zeros_like(within)
        out[sy, sx] = True
        queue = [(sy, sx)]
        taken = 1
        head = 0
        while head < len(queue) and taken < budget:
            cy, cx = queue[head]
            head += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and within[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    taken += 1
                    queue.append((ny, nx))
                    if taken >= budget:
                        break
        return out

    @staticmethod
    def _nearest_within_budget(within, seed, budget):
        """The `budget` pixels of `within` nearest `seed` by straight line.

        Global mode's trimming: with connectivity dropped there is no path to
        measure along, so straight-line distance is all that is left.
        """
        ys, xs = np.nonzero(within)
        if ys.size <= budget:
            return within
        d2 = (ys - seed[0]) ** 2 + (xs - seed[1]) ** 2
        keep = np.argpartition(d2, budget - 1)[:budget]
        out = np.zeros_like(within)
        out[ys[keep], xs[keep]] = True
        return out

    def _wand_region(self, field, seed, tolerance, negatives, is_global, budget):
        """One sample's contribution: (trimmed region, untrimmed region, capped).

        Both regions are returned from the one pass because the UI wants to say
        how hard the budget is biting, and recomputing the untrimmed region to
        answer that would double the cost of every click.
        """
        image = self._image
        within = (field <= tolerance) & image.valid_mask
        for neg in negatives:
            # Nearest-neighbour rule: a pixel survives only while its own
            # positive sample is a better match than every negative. One
            # right-click therefore removes the offending material WHEREVER IT
            # APPEARS, not only where clicked -- necessary because under global
            # mode a bleed can be hundreds of scattered fragments.
            within &= field < neg
        within[seed] = True  # a click is never a silent no-op
        region = within if is_global else self._connected_region(within, seed)
        if budget is None or int(region.sum()) <= budget:
            return region, region, False
        trim = self._nearest_within_budget if is_global else self._grow_within_budget
        return trim(region, seed, budget), region, True

    def wand_select(self, samples, options):
        """Select pixels of similar material around one or more clicks.

        samples: [[x, y], ...] positive clicks in image pixels (at least one).
        options: a single object rather than positional arguments, so a new knob
        is one key instead of another argument whose position nobody remembers.

          tolerance    float, REQUIRED. In the units of the chosen source.
          source       "image" (default, spectral) or "sam" (encoder features).
          level        "fine" (default) / "mid" / "deep". Ignored unless sam.
          max_pixels   int or None (default None = unbounded). PER SAMPLE.
          sample_size  int, default 1, clamped to 1..MAX_SAMPLE_PIXELS.
          negatives    [[x, y], ...] or None. Exclusion clicks.
          global       bool, default false. Drop the connectivity bound.
          protect      bool, default false. Skip already-labeled pixels.
                       The CALLER decides when this applies: it must not be set
                       while the eraser is active, since the eraser only ever
                       acts on labeled pixels and protecting them makes it a
                       no-op.

        Returns the selection as a bit-packed mask plus the counts the UI needs
        to explain it. Every bad input comes back as {"ok": False, "error": ...}
        rather than raising, because a traceback across the pywebview bridge is
        not something the pane can show.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        image = self._image
        opts = options if isinstance(options, dict) else {}
        try:
            points = [(int(round(float(p[1]))), int(round(float(p[0]))))
                      for p in (samples or [])]
            negatives = [(int(round(float(p[1]))), int(round(float(p[0]))))
                         for p in (opts.get("negatives") or [])]
        except (TypeError, ValueError, IndexError):
            return {"ok": False, "error": "Invalid sample coordinates"}
        if not points:
            return {"ok": False, "error": "No samples given"}
        clamp = lambda p: (min(image.height - 1, max(0, p[0])),
                           min(image.width - 1, max(0, p[1])))
        points = [clamp(p) for p in points]
        negatives = [clamp(p) for p in negatives]
        try:
            tolerance = float(opts["tolerance"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "A numeric tolerance is required"}
        try:
            sample_size = int(opts.get("sample_size", 1))
        except (TypeError, ValueError):
            return {"ok": False, "error": "sample_size must be an integer"}
        sample_size = min(MAX_SAMPLE_PIXELS, max(1, sample_size))
        budget = opts.get("max_pixels")
        if budget is not None:
            try:
                budget = int(budget)
            except (TypeError, ValueError):
                return {"ok": False, "error": "max_pixels must be an integer or null"}
            if budget <= 0:
                return {"ok": False, "error": "max_pixels must be positive"}
        source = opts.get("source", "image")
        if source not in ("image", "sam"):
            return {"ok": False, "error": f"Unknown wand source: {source}"}
        level = opts.get("level", "fine")
        if level not in FEATURE_LEVELS:
            return {"ok": False, "error": f"Unknown feature level: {level}"}
        is_global = bool(opts.get("global"))
        protect = bool(opts.get("protect"))

        if source == "sam":
            # Forced before the sample loop so a model error comes back as a
            # message the pane can show rather than a traceback on the bridge.
            reason = self._sam.unavailable_reason()
            if reason is not None:
                return {"ok": False, "error": reason}
            try:
                rgb = image.rgb_composite(self._display_bands())
                self._sam.set_image(rgb, self._sam_image_key())
            except Exception as e:
                traceback.print_exc()
                return {"ok": False, "error": f"SAM2 failed: {e}"}

        with self._lock:
            try:
                fields = self._wand_fields(points, sample_size, source, level)
                neg_fields = [f for _, f in
                              self._wand_fields(negatives, sample_size, source, level)]
            except Exception as e:
                traceback.print_exc()
                return {"ok": False, "error": f"Wand failed: {e}"}

            selected = np.zeros((image.height, image.width), dtype=bool)
            available = np.zeros_like(selected)
            capped = False
            for seed, field in fields:
                # Each negative is compared at ITS OWN distance to this pixel,
                # so the threshold a pixel must beat is the negative's field
                # value there -- not a scalar. neg_fields are full fields.
                region, uncapped, hit = self._wand_region(
                    field, seed, tolerance, neg_fields, is_global, budget)
                selected |= region
                available |= uncapped
                capped = capped or hit

        # Protection is applied to the FINISHED selection, not the match set: a
        # labeled pixel should still conduct the fill through itself, it simply
        # does not get painted. Masking earlier would sever a region at the
        # first labeled pixel it touches, so a fill could not reach unlabeled
        # ground on the far side of a labeled strip.
        protected = 0
        if protect and self._labels is not None:
            labeled = selected & (self._labels != UNLABELED)
            protected = int(labeled.sum())
            selected &= ~labeled
        return {
            "ok": True,
            "mask": base64.b64encode(np.packbits(selected).tobytes()).decode("ascii"),
            "count": int(selected.sum()),
            "available": int(available.sum()),
            "protected": protected,
            "capped": capped,
            "valid": int(image.valid_mask.sum()),
            "width": image.width,
            "height": image.height,
        }

    # ---------- training ----------

    def _labeled_samples(self, on_progress=None):
        """(name, role, loader) for every labeled image in the project.

        One disk scan shared by training and the accuracy report, so the two can
        never disagree about which labels exist — the whole point of the report
        is the gap between the two roles, which is meaningless if they are
        measured over different ground truth.

        role is "training" or "validation" (project.image_role); a standalone
        image with no project is always "training". The active image contributes
        its in-memory labels (which may not be saved yet); the others are taken
        from masks_user/. Loaders are lazy so callers hold one image at a time
        instead of the whole project.

        Each candidate costs a mask read plus a header open, so on a large
        project this runs for many seconds; on_progress(done, total) is called
        per image so a caller on a worker thread can report the wait instead of
        appearing hung. It is notification only -- it cannot abort the scan.
        """
        if self._image is None:
            return []
        active = []
        if (self._labels != UNLABELED).any():
            active = [(self._image_name, "training",
                       lambda: (self._image, self._labels))]
        if self._project is None:  # standalone image: no project, so no split
            return active
        if active:  # the active image's role comes from the split like any other
            active = [(self._image_name, self._project.image_role(self._image_name),
                       active[0][2])]
        samples = []
        names = self._project.list_images()
        for done, name in enumerate(names):
            if on_progress is not None:
                on_progress(done, len(names))
            if name == self._image_name:
                samples.extend(active)
                continue
            user_path, _ = self._project.mask_paths(name)
            mask = self._read_mask_file(user_path)
            if mask is None or not (mask != UNLABELED).any():
                continue
            if not self._mask_fits_image(name, mask):
                continue  # stale mask from a resized/replaced image
            samples.append((name, self._project.image_role(name),
                            self._make_sample_loader(name, mask)))
        if on_progress is not None:
            on_progress(len(names), len(names))
        return samples

    def _mask_fits_image(self, name, mask):
        """Cheap header-only check that a saved mask still matches its image."""
        try:
            with rasterio.open(self._project.image_path(name)) as src:
                return ((src.height, src.width) == mask.shape
                        and src.count == self._project.input_channels)
        except Exception:
            return False

    def _make_sample_loader(self, name, mask):
        def load():
            image = SentinelImage(self._project.image_path(name),
                                  expected_bands=self._project.input_channels,
                                  patch_size=self._project.patch_size)
            return image, mask
        return load

    def start_training(self, epochs):
        """Validate cheaply, then hand the whole run to a worker.

        Only the guards that cost nothing are answered here. Collecting the
        samples reads and shape-checks every mask in the project, which on a
        large project runs for many seconds, so it happens on the worker with
        the status already reading "training" -- otherwise the UI sits on an
        idle-looking toolbar for the whole scan. The cost of that is that "you
        have not labeled anything" can no longer be returned from this call; it
        arrives through the status like any other training failure.
        """
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Training already running"}
        if self._report["running"]:
            return {"ok": False, "error": "Wait for the accuracy report to finish"}
        # Same clamp train() applies, so the reported total is right from the
        # start rather than being corrected by the first progress callback. Only
        # a floor of 1 -- the selected length is otherwise run as asked, and is
        # returned below so the caller can show what is actually being trained.
        epochs = max(1, int(epochs))
        self._stop_event.clear()
        self._status.update({"state": "training", "stage": "collecting",
                             "epoch": 0, "epochs": epochs, "images": 0,
                             "scan_done": 0, "scan_total": 0, "error": None})
        threading.Thread(target=self._train_worker, args=(epochs,),
                         daemon=True).start()
        return {"ok": True, "epochs": epochs}

    def stop_training(self):
        self._stop_event.set()
        return {"ok": True}

    def _on_scan(self, done, total):
        self._status.update({"scan_done": done, "scan_total": total})

    def _train_worker(self, epochs):
        try:
            labeled = self._labeled_samples(on_progress=self._on_scan)
            # Validation images are held out entirely -- not merely their labels
            # -- so the accuracy report scores the model on imagery it has never
            # seen.
            samples = [(name, load) for name, role, load in labeled
                       if role == "training"]
            if not samples:
                # Distinguish "nothing labeled" from "everything labeled is held
                # out", which is otherwise a baffling place to land.
                if labeled:
                    self._fail_training(
                        f"All {len(labeled)} labeled image"
                        f"{'s are' if len(labeled) != 1 else ' is'} in the "
                        f"validation set; right-click a thumbnail to move "
                        f"one to training")
                else:
                    self._fail_training("Label at least one image before training")
                return
            # The scan is not interruptible, so a Stop pressed during it lands
            # here rather than being lost.
            if self._stop_event.is_set():
                self._status.update({"state": "idle", "stage": None})
                return
            self._status.update({"stage": "running", "images": len(samples)})
            with self._lock:
                done = train_module.train(
                    self._model, [load for _, load in samples], epochs,
                    progress=self._on_progress,
                    stop_event=self._stop_event,
                )
                self._status["total_epochs"] += done
                self._model_version += 1
                self._save_model_checkpoint()
                self._class_map, self._probs = self._model.predict_image(self._image)
            self._status.update({"state": "idle", "stage": None})
        except Exception:
            self._status.update({"state": "error", "stage": None,
                                 "error": traceback.format_exc(limit=3)})

    def _fail_training(self, message):
        """End the run with a message meant for the user, not a traceback."""
        self._status.update({"state": "error", "stage": None, "error": message})

    def _on_progress(self, stats):
        self._status.update({
            "epoch": stats["epoch"],
            "epochs": stats["epochs"],
            "loss_sup": round(stats["loss_sup"], 4),
            "loss_pseudo": round(stats["loss_pseudo"], 4),
            "loss_cons": round(stats["loss_cons"], 4),
            "iou": stats["iou"],  # live target IoU, or None if unscoreable
            # 0.0 while the unsupervised gate is still shut. Surfaced because
            # the gate opens on measured IoU rather than on a fixed epoch, so
            # without it a run of zero pseudo/cons losses is unreadable.
            "ramp": stats["ramp"],
        })

    def get_status(self):
        return dict(self._status)

    def _save_model_checkpoint(self):
        """Persist the project's trained model (weights + epoch counter) atomically."""
        if self._project is None:
            return
        path = self._project.model_checkpoint_path
        tmp = path + ".tmp"
        torch.save({
            "state_dict": self._model.net.state_dict(),
            "in_channels": self._model.in_channels,
            "use_pointrend": self._model.use_pointrend,
            "total_epochs": self._status["total_epochs"],
        }, tmp)
        os.replace(tmp, path)

    @staticmethod
    def _load_model_checkpoint(model, project):
        """Restore the project's model weights; returns the saved epoch count.

        Unreadable or profile-mismatched checkpoints are ignored (the model
        stays freshly initialized), like the corrupt-mask fallback in
        _read_user_mask. Loading is filtered by name and shape: toggling
        PointRend adds/removes the point head, and the shared encoder/decoder
        weights must survive the toggle in either direction. The same filter
        discards checkpoints written by an older architecture.
        """
        path = project.model_checkpoint_path
        if not os.path.exists(path):
            return 0
        try:
            ckpt = torch.load(path, map_location=model.device, weights_only=True)
            if ckpt.get("in_channels") != model.in_channels:
                return 0
            model_state = model.net.state_dict()
            state = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if k in model_state and v.shape == model_state[k].shape
            }
            model.net.load_state_dict(state, strict=False)
            return int(ckpt.get("total_epochs", 0))
        except Exception:
            return 0

    def reset_model(self):
        """Discard all training and reinitialize from the default pretrained weights."""
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._report["running"]:
            return {"ok": False, "error": "Wait for the accuracy report to finish"}
        with self._lock:
            self._model = SegmentationModel(in_channels=self._model.in_channels,
                                            patch_size=self._model.patch_size,
                                            use_pointrend=self._model.use_pointrend)
            self._model_version += 1
            self._class_map = None
            self._probs = None
            if self._project is not None:
                # otherwise reopening the project resurrects the discarded model
                try:
                    os.remove(self._project.model_checkpoint_path)
                except FileNotFoundError:
                    pass
        self._status.update({"state": "idle", "epoch": 0, "epochs": 0, "total_epochs": 0,
                             "images": 0, "ramp": None, "stage": None,
                             "scan_done": 0, "scan_total": 0,
                             "loss_sup": None, "loss_pseudo": None, "loss_cons": None, "error": None})
        return {"ok": True}

    # ---------- project ----------

    def new_project(self):
        """Pick a directory; open its existing project or propose setup defaults.

        If the folder has no project_config.json yet, nothing is written:
        the frontend shows the setup dialog with the returned defaults and
        calls create_project() once the user confirms.
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        root = self._folder_dialog()
        if not root:
            return {"ok": False, "error": None}  # user cancelled
        if os.path.exists(config_path(root)):
            return self._init_project(root)
        channels = probe_band_count(root)
        config = default_config(os.path.basename(os.path.normpath(root)) or "project",
                                input_channels=channels or 10)
        return {"ok": True, "needs_setup": True, "root": root,
                "defaults": {
                    "project_name": config["project_name"],
                    "input_channels": config["data_profile"]["input_channels"],
                    "input_patch_size": config["data_profile"]["input_patch_size"][0],
                    "band_names": config["data_profile"]["band_names"],
                    "display_bands": config["data_profile"]["display_bands"],
                    "use_pointrend": config["data_profile"]["use_pointrend"],
                    "validation_ratio": config["split"]["validation_ratio"],
                    "images_folder": config["paths"]["images_folder"],
                    "masks_user": config["paths"]["masks_user"],
                    "masks_ai": config["paths"]["masks_ai"],
                    "bands_detected": channels is not None,
                }}

    def new_project_from_geotiff(self):
        """Pick one GeoTIFF and propose a project built by tiling it.

        Nothing is written here: the band profile is read off the raster and
        returned as setup defaults (alongside tile size / overlap), and
        `create_project_from_geotiff` does the work once the user confirms.
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        source = self._file_dialog()
        if not source:
            return {"ok": False, "error": None}  # user cancelled
        try:
            with rasterio.open(source) as src:
                channels, width, height = src.count, src.width, src.height
                descriptions = [d for d in (src.descriptions or []) if d]
                has_crs = src.crs is not None
        except Exception as e:
            return {"ok": False, "error": f"Could not read that GeoTIFF: {e}"}

        stem = os.path.splitext(os.path.basename(source))[0]
        config = default_config(stem or "project", input_channels=channels)
        patch = config["data_profile"]["input_patch_size"][0]
        # Default tiles comfortably larger than the model patch (a tile smaller
        # than the patch is unusable), clamped to the raster itself.
        tile = max(patch, min(256, width, height))
        band_names = (list(descriptions) if len(descriptions) == channels
                      else config["data_profile"]["band_names"])
        return {"ok": True, "needs_setup": True, "mode": "geotiff",
                "source": source,
                "defaults": {
                    "project_name": config["project_name"],
                    "input_channels": channels,
                    "input_patch_size": patch,
                    "band_names": band_names,
                    "display_bands": config["data_profile"]["display_bands"],
                    "use_pointrend": config["data_profile"]["use_pointrend"],
                    "validation_ratio": config["split"]["validation_ratio"],
                    "images_folder": "images",
                    "masks_user": config["paths"]["masks_user"],
                    "masks_ai": config["paths"]["masks_ai"],
                    "bands_detected": True,
                    "output_root": self._suggest_project_dir(source, stem),
                    "tile_width": tile,
                    "tile_height": tile,
                    "overlap": 0,
                    "source_name": os.path.basename(source),
                    "source_width": width,
                    "source_height": height,
                    "source_georeferenced": has_crs,
                }}

    @staticmethod
    def _suggest_project_dir(source, stem):
        """A not-yet-existing sibling folder to tile into."""
        parent = os.path.dirname(os.path.abspath(source))
        base = os.path.join(parent, f"{stem}_project")
        candidate, n = base, 2
        while os.path.exists(candidate):
            candidate = f"{base}_{n}"
            n += 1
        return candidate

    def create_project_from_geotiff(self, source, settings):
        """Tile `source` into a new folder and open it as a project."""
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        root = str(settings.get("output_root", "")).strip()
        if not root:
            return {"ok": False, "error": "Choose an output folder"}
        root = os.path.abspath(os.path.expanduser(root))
        # Never tile into a folder that already holds something: this writes
        # many files and would be tedious to disentangle from existing data.
        if os.path.exists(config_path(root)):
            return {"ok": False, "error": "That folder already contains a project"}
        if os.path.isdir(root) and os.listdir(root):
            return {"ok": False, "error": "That folder is not empty; pick a new one"}
        if os.path.isfile(root):
            return {"ok": False, "error": "That path is a file, not a folder"}

        try:
            tile_w = int(settings["tile_width"])
            tile_h = int(settings["tile_height"])
            overlap = int(settings.get("overlap") or 0)
            patch = int(settings["input_patch_size"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "Tile size and overlap must be whole numbers"}
        # SentinelImage refuses an image smaller than the patch, so tiles that
        # small would produce a project whose every image fails to open.
        if min(tile_w, tile_h) < patch:
            return {"ok": False,
                    "error": f"Tile size ({tile_w}x{tile_h}) must be at least the "
                             f"patch size ({patch} px)"}

        if self._tiling["running"]:
            return {"ok": False, "error": "Already tiling"}
        images_folder = str(settings.get("images_folder", "")).strip() or "images"

        # Everything above is validated synchronously so mistakes come back
        # immediately. The tiling itself can run for minutes on a big raster,
        # so it moves to a worker and the dialog polls tiling_progress().
        self._tiling = {"running": True, "done": 0, "total": 0,
                        "result": None, "error": None}
        threading.Thread(
            target=self._tile_worker,
            args=(source, root, images_folder, tile_w, tile_h, overlap, settings),
            daemon=True).start()
        return {"ok": True, "started": True}

    def _tile_worker(self, source, root, images_folder, tile_w, tile_h, overlap, settings):
        """Tile, write the config and open the project; report via self._tiling."""
        try:
            def on_progress(done, total):
                self._tiling["done"] = done
                self._tiling["total"] = total

            summary = tile_geotiff(source, os.path.join(root, images_folder),
                                   tile_w, tile_h, overlap, progress=on_progress)
            if not summary["tiles"]:
                raise ValueError("Tiling produced no images (every tile was empty)")
            config = config_from_settings(settings)
            save_config(root, config)
            result = self._init_project(root)
            if result.get("ok"):
                result["created"] = True
                result["tiling"] = summary
            self._tiling["result"] = result
        except Exception as e:  # a worker crash must surface in the dialog
            traceback.print_exc()
            self._tiling["error"] = str(e)
        finally:
            self._tiling["running"] = False

    def tiling_progress(self):
        """Poll the background tiling: counts while running, payload when done."""
        t = self._tiling
        return {"running": t["running"], "done": t["done"], "total": t["total"],
                "result": t["result"], "error": t["error"]}

    def pick_folder(self):
        """Folder chooser for the setup dialog's output path."""
        return {"path": self._folder_dialog()}

    def create_project(self, root, settings):
        """Write the config assembled in the setup dialog and open the project."""
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if os.path.exists(config_path(root)):
            return {"ok": False, "error": "This folder already contains a project"}
        try:
            config = config_from_settings(settings)
            save_config(root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        result = self._init_project(root)
        if result.get("ok"):
            result["created"] = True
        return result

    def probe_bands(self, root, images_folder="."):
        """Band count of the images in root/images_folder (None if unreadable)."""
        folder = os.path.normpath(os.path.join(root, images_folder))
        return {"bands": probe_band_count(folder)}

    def _init_project(self, root):
        """Open (or create) the project at root, kick off thumbnails, load image 1.

        The model is rebuilt to the project's data_profile (input_channels,
        input_patch_size), so 3/4-band projects get a matching network.
        """
        try:
            if os.path.exists(config_path(root)):
                project = Project.open(root)
                created = False
            else:
                project = Project.create(root)
                save_config(root, project.config)
                created = True
            model = SegmentationModel(in_channels=project.input_channels,
                                      patch_size=project.patch_size,
                                      use_pointrend=project.use_pointrend)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        restored_epochs = self._load_model_checkpoint(model, project)
        with self._lock:
            self._autosave_user_mask()  # labels of the previously open project
            self._set_image(None, None)
            self._model = model
            self._model_version += 1
            self._dirty_labels = {}
            self._dirty_geo = {}
        self._status.update({"state": "idle", "epoch": 0, "epochs": 0,
                             "total_epochs": restored_epochs, "images": 0, "ramp": None,
                             "stage": None, "scan_done": 0, "scan_total": 0,
                             "loss_sup": None, "loss_pseudo": None, "loss_cons": None,
                             "error": None})
        self._project = project
        project.start_thumbnail_generation()
        images = project.list_images()
        image_error = None
        if images:
            result = self.load_image(images[0])
            image_error = result.get("error")
        return {"ok": True, "created": created, "project": self.get_project(),
                "image": self._image_payload(), "image_error": image_error}

    def save_project(self):
        """Write/update project_config.json in the project root."""
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        try:
            path = save_config(self._project.root, self._project.config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "path": path}

    def update_settings(self, settings):
        """Apply edited settings to the open project's config and persist them.

        Band count and patch size define the model and are fixed at creation,
        so they are preserved here; only the project name, band names, folder
        paths and the PointRend toggle change. Changing a folder path reopens
        the project so the new folders take effect (mask dirs are created and
        images reload); the model checkpoint restores because the architecture
        is unchanged. Toggling PointRend rebuilds the model with/without the
        point head and re-restores the checkpoint (filtered, so the trained
        U-Net weights survive the toggle).
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._report["running"]:
            return {"ok": False, "error": "Wait for the accuracy report to finish"}
        old = self._project.config
        profile = old["data_profile"]
        merged = dict(settings)
        merged["input_channels"] = profile["input_channels"]
        merged["input_patch_size"] = profile["input_patch_size"][0]
        try:
            config = config_from_settings(merged)
            # config_from_settings resets the palette to the defaults; keep the
            # project's ids/names and only accept edited colors from the dialog
            # (validated below by save_config -> validate_config).
            new_classes = settings.get("classes")
            by_id = {c["id"]: c for c in old["classes"]}
            merged_classes = None
            if isinstance(new_classes, list) and len(new_classes) == len(old["classes"]):
                merged_classes = []
                for nc in new_classes:
                    if not isinstance(nc, dict) or nc.get("id") not in by_id:
                        merged_classes = None
                        break
                    cls = dict(by_id[nc["id"]])
                    if isinstance(nc.get("color"), str):
                        cls["color"] = nc["color"]
                    merged_classes.append(cls)
            config["classes"] = merged_classes or [dict(c) for c in old["classes"]]
            # config_from_settings rebuilds the whole dict, so anything not
            # carried over here is silently dropped. The split is creation-time
            # (ratio) plus the user's per-image moves (overrides): neither is
            # editable from this dialog, so both survive verbatim.
            config["split"] = dict(old.get("split") or default_split())
            save_config(self._project.root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        if config["paths"] != old["paths"]:
            result = self._init_project(self._project.root)
            result["reopened"] = True
            return result
        self._project.config = config
        result = {"ok": True, "project": self.get_project()}
        if (config["data_profile"]["use_pointrend"]
                != bool(profile.get("use_pointrend", False))):
            # The point head is part of the network: rebuild the model with the
            # new flag and re-restore the checkpoint (filtered, so the trained
            # U-Net weights survive; a fresh/dropped point head is expected).
            with self._lock:
                model = SegmentationModel(in_channels=self._model.in_channels,
                                          patch_size=self._model.patch_size,
                                          use_pointrend=self._project.use_pointrend)
                self._status["total_epochs"] = self._load_model_checkpoint(
                    model, self._project)
                self._model = model
                self._model_version += 1
                self._class_map = None
                self._probs = None
        if config["data_profile"]["display_bands"] != profile.get("display_bands"):
            # New RGB mapping: re-render the open image (labels/geometry unchanged,
            # so the frontend swaps only the base png) and rebuild every cached
            # thumbnail, overwriting the old ones.
            if self._image is not None:
                result["image"] = self._image_payload()
            # The SAM2 encoder reads exactly those three bands, so every cached
            # embedding field now describes an image that no longer exists. The
            # spectral fields are dropped with them: one cache, one lifetime.
            self._clear_wand_cache()
            self._project.start_thumbnail_generation(force=True)
        return result

    def get_project(self):
        """Current project state for the frontend, or None."""
        if self._project is None:
            return None
        return {
            "root": self._project.root,
            "name": self._project.config["project_name"],
            "classes": self._project.config["classes"],
            "data_profile": self._project.config["data_profile"],
            "paths": self._project.config["paths"],
            "images": self._project.list_images(),
            "validation": self._project.validation_names(),
            "validation_ratio": self._project.validation_ratio,
            "active": self._image_name,
        }

    def set_image_role(self, name, role):
        """Move one image between the training and validation sets.

        The split is computed from the ratio, so only a deviation from that
        computed assignment is persisted; moving an image back to where it would
        have landed anyway drops the override again and keeps the config tidy.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if role not in SPLIT_ROLES:
            return {"ok": False, "error": f"Unknown role: {role}"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._report["running"]:
            return {"ok": False, "error": "Wait for the accuracy report to finish"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}
        config = self._project.config
        split = dict(config.get("split") or default_split())
        overrides = dict(split.get("overrides") or {})
        names = self._project.list_images()
        base = "validation" if name in deterministic_validation(
            names, split.get("validation_ratio", 0.2)) else "training"
        if role == base:
            overrides.pop(name, None)
        else:
            overrides[name] = role
        split["overrides"] = overrides
        config["split"] = split
        try:
            save_config(self._project.root, config)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "project": self.get_project()}

    def delete_image(self, name):
        """Permanently remove an image and its user/AI masks and thumbnail.

        If the deleted image is the active one, its in-memory buffer is
        dropped without autosaving (there is nothing left on disk to save it
        against) and the next remaining image is loaded in its place, or the
        panes clear if none are left.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._report["running"]:
            return {"ok": False, "error": "Wait for the accuracy report to finish"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}

        was_active = name == self._image_name
        user_path, ai_path = self._project.mask_paths(name)
        for path in (self._project.image_path(name), user_path, ai_path,
                     self._project.thumb_path(name)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                return {"ok": False, "error": f"Could not delete {name}: {e}"}

        self._dirty_labels.pop(name, None)
        self._dirty_geo.pop(name, None)

        # A cached accuracy report may still list the deleted image: drop its
        # row and re-pool the set summaries from what remains, rather than
        # invalidating the whole cache (deleting an image doesn't retrain, so
        # every other row's score is still exactly right).
        cached = self._report["result"]
        if cached is not None:
            rows = [r for r in cached["rows"] if r["name"] != name]
            if len(rows) != len(cached["rows"]):
                self._report["result"] = report_module.build_report(
                    rows, total_epochs=cached["total_epochs"],
                    project_name=cached["project_name"],
                    generated_at=cached["generated_at"])

        # Drop a manual split override for the deleted name so the config
        # doesn't keep pointing at an image that no longer exists.
        config = self._project.config
        split = dict(config.get("split") or default_split())
        overrides = dict(split.get("overrides") or {})
        if overrides.pop(name, None) is not None:
            split["overrides"] = overrides
            config["split"] = split
            try:
                save_config(self._project.root, config)
            except (OSError, ValueError) as e:
                return {"ok": False, "error": str(e)}

        result = {"ok": True}
        if was_active:
            with self._lock:
                self._set_image(None, None)  # the buffer's file is already gone
            remaining = self._project.list_images()
            if remaining:
                loaded = self.load_image(remaining[0])
                result["image"] = loaded.get("image")
                if loaded.get("error"):
                    result["image_error"] = loaded["error"]
            else:
                result["image"] = None
        result["project"] = self.get_project()
        return result

    def load_image(self, name):
        """Make an image of the project active in both panes.

        The current user labels are autosaved to masks_user/ first, and the
        new image's saved user mask (if any) is restored as its labels.
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if name not in self._project.list_images():
            return {"ok": False, "error": f"Unknown image: {name}"}
        if name == self._image_name:
            return {"ok": True, "image": self._image_payload()}
        try:
            image = SentinelImage(self._project.image_path(name),
                                  expected_bands=self._project.input_channels,
                                  patch_size=self._project.patch_size)
        except Exception as e:
            return {"ok": False, "error": f"Could not load {name}: {e}"}
        with self._lock:
            self._autosave_user_mask()
            self._set_image(image, name, labels=self._read_user_mask(image, name))
        return {"ok": True, "image": self._image_payload()}

    def _autosave_user_mask(self):
        """Persist current labels before switching images so no work is lost."""
        if self._project is None or self._image is None:
            return
        user_path, _ = self._project.mask_paths(self._image_name)
        if (self._labels != UNLABELED).any() or os.path.exists(user_path):
            write_mask_geotiff(user_path, self._labels, self._image.crs, self._image.transform)

    @staticmethod
    def _read_mask_file(path):
        """A saved mask as (H, W) uint8, or None if missing/unreadable.

        Corrupt masks read as None so callers fall back to unlabeled rather
        than failing an image switch or a training run.
        """
        if not os.path.exists(path):
            return None
        try:
            with rasterio.open(path) as src:
                return src.read(1).astype(np.uint8)
        except Exception:
            return None

    def _read_user_mask(self, image, name):
        """Labels for `name`: this session's edits if any, else the saved mask_user/ file.

        Edits made earlier in the session (possibly not yet written to disk by
        an explicit Save) take priority, so switching back to an image always
        shows its latest state instead of only what's on disk.
        """
        cached = self._dirty_labels.get(name)
        if cached is not None and cached.shape == (image.height, image.width):
            return cached.copy()
        user_path, _ = self._project.mask_paths(name)
        mask = self._read_mask_file(user_path)
        if mask is not None and mask.shape == (image.height, image.width):
            return mask
        return None

    # ---------- masks ----------

    def autosave_user_mask(self):
        """Persist the current image's labels to masks_user/ after an edit.

        Writes only the active image (cheap enough to run per stroke) and,
        unlike save_user_mask, leaves the session dirty-tracking untouched so
        the manual "Save Masks" still writes every edited image at once.
        """
        if self._project is None or self._image is None:
            return {"ok": False}
        with self._lock:
            self._autosave_user_mask()
        return {"ok": True}

    def save_user_mask(self):
        """Save every user-labeled image edited this session to masks_user/ as GeoTIFF.

        Tracks per-image label buffers in self._dirty_labels as the user edits
        (set_labels/transfer_prediction), so switching between images during a
        session and then saving once still writes all of them, not just the
        currently active image.
        """
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        if self._image_name and self._image is not None:
            self._dirty_labels[self._image_name] = self._labels
            self._dirty_geo[self._image_name] = (self._image.crs, self._image.transform)
        if not self._dirty_labels:
            return {"ok": False, "error": "No edited masks to save"}
        with self._lock:
            paths = []
            for name, labels in self._dirty_labels.items():
                user_path, _ = self._project.mask_paths(name)
                crs, transform = self._dirty_geo[name]
                write_mask_geotiff(user_path, labels, crs, transform)
                paths.append(user_path)
            self._dirty_labels.clear()
            self._dirty_geo.clear()
        return {"ok": True, "path": paths[0] if len(paths) == 1 else f"{len(paths)} masks"}

    def save_ai_mask(self):
        """Left pane: save the model inference to masks_ai/ as GeoTIFF."""
        if self._project is None or self._image is None:
            return {"ok": False, "error": "No project image loaded"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        _, ai_path = self._project.mask_paths(self._image_name)
        with self._lock:
            if self._class_map is None:
                self._class_map, self._probs = self._model.predict_image(self._image)
            mask = self._class_map.copy()
            mask[~self._image.valid_mask] = UNLABELED
            write_mask_geotiff(ai_path, mask, self._image.crs, self._image.transform)
        return {"ok": True, "path": ai_path}

    # ---------- thumbnails ----------

    def generate_thumbnails(self):
        """(Re)start async generation of missing 128px thumbnails in .thumbnails/."""
        if self._project is None:
            return {"ok": False, "error": "No project open"}
        self._project.start_thumbnail_generation()
        return {"ok": True}

    def get_thumbnails(self):
        """Thumbnail PNGs (base64, None while pending) plus generation progress."""
        if self._project is None:
            return {"running": False, "done": 0, "total": 0, "thumbs": {}}
        thumbs = {}
        for name in self._project.list_images():
            path = self._project.thumb_path(name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    thumbs[name] = base64.b64encode(f.read()).decode("ascii")
            else:
                thumbs[name] = None
        return {**self._project.thumb_progress, "thumbs": thumbs}

    # ---------- export ----------

    def export_model(self):
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        path = self._save_dialog("model.pth")
        if not path:
            return {"ok": False, "error": None}  # user cancelled
        with self._lock:
            self._model.save(path)
        return {"ok": True, "path": path}

    def export_onnx(self):
        """Export the U-Net to ONNX for portable inference.

        Graph: input (N, C, H, W) float32 of per-band normalized bands (the same
        [0,1] robust-percentile normalization used in training) -> logits
        (N, 1, H, W); sigmoid(logit) = P(target), so class 0 (target) is the
        positive class and 1 (background) its complement. Batch and spatial dims
        are dynamic (H and W must stay multiples of 32, the encoder's stride).
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        path = self._save_dialog("model.onnx")
        if not path:
            return {"ok": False, "error": None}  # user cancelled
        try:
            self._write_model_onnx(path)
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"ONNX export failed: {e}"}
        return {"ok": True, "path": path}

    def _write_model_onnx(self, path):
        """Export the network to `path` as ONNX (+ metadata). Shared by exports.

        Only the batch axis is dynamic. The bottleneck's positional embedding is
        sized for exactly one patch size, and both consumers (predict_image and
        the standalone predictor) only ever feed patch_size tiles, so declaring
        the spatial axes dynamic would only advertise sizes the graph cannot
        actually serve.
        """
        with self._lock:
            net = self._model.net
            net.eval()
            channels, size = self._model.in_channels, self._model.patch_size
            dummy = torch.zeros(1, channels, size, size, device=self._model.device)
            torch.onnx.export(
                net, dummy, path,
                input_names=["input"], output_names=["logits"],
                opset_version=17, dynamo=False,
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        self._annotate_onnx(path)

    def _annotate_onnx(self, path):
        """Best-effort: embed model metadata into the exported ONNX file."""
        try:
            import json

            import onnx

            profile = self._project.config["data_profile"] if self._project else {}
            model = onnx.load(path)
            onnx.helper.set_model_props(model, {
                "in_channels": str(self._model.in_channels),
                "patch_size": str(self._model.patch_size),
                "use_pointrend": str(self._model.use_pointrend),
                "band_names": json.dumps(profile.get("band_names", [])),
                "output": "logits; softmax over 2 channels; classes 0=target 1=background",
                "input_normalization": "per-band 2-98 percentile robust scaling to [0,1]",
            })
            onnx.save(model, path)
        except Exception:
            traceback.print_exc()  # metadata is optional; the .onnx is already written

    def executable_available(self):
        """Whether the prebuilt standalone predictor is present to export."""
        exe = os.path.join(_PREDICTOR_DIR, "predictor")
        ok = os.path.isdir(_PREDICTOR_DIR) and (
            os.path.exists(exe) or os.path.exists(exe + ".exe"))
        return {"available": ok,
                "reason": None if ok else "Predictor not built (run standalone/predictor.spec)"}

    def export_executable(self):
        """Export a clickable standalone predictor.

        Writes a self-contained folder: the prebuilt predictor executable plus
        this project's model as `model.onnx` beside it (the exe loads the sibling
        model at runtime). The user runs the executable, picks a GeoTIFF, maps
        its bands to the model channels, and gets a prediction GeoTIFF — all
        without Python installed.
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if not self.executable_available()["available"]:
            return {"ok": False,
                    "error": "Predictor executable not built; run standalone/predictor.spec"}
        folder = self._folder_dialog()
        if not folder:
            return {"ok": False, "error": None}  # user cancelled
        import shutil

        dest = os.path.join(folder, "UNet-Predictor")
        try:
            shutil.copytree(_PREDICTOR_DIR, dest, dirs_exist_ok=True)
            self._write_model_onnx(os.path.join(dest, "model.onnx"))
        except (OSError, RuntimeError) as e:
            traceback.print_exc()
            return {"ok": False, "error": f"Export failed: {e}"}
        return {"ok": True, "path": dest}

    # ---------- accuracy report ----------

    def generate_accuracy_report(self):
        """Score the model on the validation and training images, in background.

        Runs over every labeled image, so it takes minutes on a large project and
        follows the tiling pattern: validate synchronously, hand an immutable
        snapshot to a worker, report through accuracy_report_progress().

        The last finished report is reused as-is when nothing has retrained or
        replaced the model since (see _model_version) -- rescoring every image
        again would produce byte-identical numbers, so accuracy_report_progress
        is left pointing at the existing result instead of a new run.
        """
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        if self._report["running"]:
            return {"ok": False, "error": "A report is already running"}
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        if (self._report["result"] is not None
                and self._report.get("model_version") == self._model_version):
            return {"ok": True, "started": False, "cached": True}
        # Snapshot on *this* thread, before the worker starts. _dirty_labels
        # stores a reference to the live label array and transfer_prediction
        # mutates it in place, so a worker holding the bare reference would
        # score labels the user is still painting. Only the dirty buffers are
        # copied (they are already resident); the rest stay lazy disk loaders.
        samples = []
        for name, role, load in self._labeled_samples():
            if name == self._image_name:
                snapshot = self._labels.copy()
                samples.append((name, role,
                                lambda img=self._image, m=snapshot: (img, m)))
            else:
                samples.append((name, role, load))
        if not samples:
            return {"ok": False,
                    "error": "Label at least one image before generating a report"}
        self._report = {"running": True, "done": 0, "total": len(samples),
                        "result": None, "error": None,
                        "model_version": self._model_version}
        self._report_stop.clear()
        threading.Thread(target=self._report_worker, args=(samples,),
                         daemon=True).start()
        return {"ok": True, "started": True, "total": len(samples)}

    def _report_worker(self, samples):
        """Predict and score each snapshotted sample; report via self._report."""
        try:
            model = self._model  # captured once: a model swap cannot split a run
            rows = []
            for name, role, load in samples:
                if self._report_stop.is_set():
                    self._report["error"] = "Cancelled"
                    return
                image, labels = load()
                # The lock is taken per image, not for the whole run: holding it
                # throughout would block load_image, get_overlay and the label
                # autosave for minutes, freezing the UI with no explanation.
                with self._lock:
                    class_map, _ = model.predict_image(image)
                rows.append(report_module.score_image(
                    name, role, labels, class_map, image.valid_mask))
                del image, class_map  # bound memory: one image resident at a time
                self._report["done"] += 1
            self._report["result"] = report_module.build_report(
                rows,
                total_epochs=self._status["total_epochs"],
                project_name=(self._project.config["project_name"]
                              if self._project else self._image_name or ""),
                generated_at=time.strftime("%Y-%m-%d %H:%M"),
            )
        except Exception as e:  # a worker crash must surface in the modal
            traceback.print_exc()
            self._report["error"] = str(e)
        finally:
            self._report["running"] = False

    def accuracy_report_progress(self):
        """Poll the background report: counts while running, payload when done."""
        r = self._report
        return {"running": r["running"], "done": r["done"], "total": r["total"],
                "result": r["result"], "error": r["error"]}

    def cancel_accuracy_report(self):
        self._report_stop.set()
        return {"ok": True}

    def save_accuracy_report(self):
        """Write the last finished report to a self-contained HTML file.

        Takes no argument: it renders the report cached by the worker, so the
        saved file provably matches what the modal displayed and no large
        payload round-trips through the JS bridge.
        """
        result = self._report["result"]
        if result is None:
            return {"ok": False, "error": "No report to save"}
        path = self._save_dialog("accuracy_report.html")
        if not path:
            return {"ok": False, "error": None}  # user cancelled
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_module.render_html(result))
        except OSError as e:
            return {"ok": False, "error": f"Could not write the report: {e}"}
        return {"ok": True, "path": path}

    def export_mask(self):
        """Model prediction merged with user labels (user pixels win), as GeoTIFF."""
        if self._image is None:
            return {"ok": False, "error": "No image loaded"}
        if self._status["state"] == "training":
            return {"ok": False, "error": "Wait for training to finish"}
        path = self._save_dialog("mask.tif")
        if not path:
            return {"ok": False, "error": None}
        with self._lock:
            class_map, self._probs = self._model.predict_image(self._image)
            self._class_map = class_map
            mask = class_map.copy()
            user = self._labels != UNLABELED
            mask[user] = self._labels[user]
            mask[~self._image.valid_mask] = UNLABELED
            write_mask_geotiff(path, mask, self._image.crs, self._image.transform)
        return {"ok": True, "path": path}

    @staticmethod
    def _save_dialog(default_name):
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        result = window.create_file_dialog(webview.FileDialog.SAVE, save_filename=default_name)
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result

    @staticmethod
    def _file_dialog():
        """Pick a single existing GeoTIFF."""
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        result = window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=False,
            file_types=("GeoTIFF (*.tif;*.tiff)", "All files (*.*)"))
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result

    @staticmethod
    def _folder_dialog():
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return None
        result = window.create_file_dialog(webview.FileDialog.FOLDER)
        if isinstance(result, (tuple, list)):
            return result[0] if result else None
        return result
