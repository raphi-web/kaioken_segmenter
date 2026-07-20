"""Efficient-SAM2 "snap-to-edge" click segmentation, running on ONNX Runtime.

The SAM2 image predictor is exported to two ONNX graphs (see
`export_sam_onnx.py`) and run here with onnxruntime, so the app needs neither
PyTorch-SAM2 nor hydra at runtime for this feature:

  * encoder graph: preprocessed 1x3x1024x1024 image -> image_embed + 2 high-res
    feature maps. Runs once per image (~seconds on CPU); the outputs are cached.
  * decoder graph: those feature maps + clicks (foreground/background) ->
    low-res mask logits + IoU. Runs per click (fast) on the cached embeddings.

Only the trivial resize/normalize (torchvision, already a core dependency) and
the final mask upscale/threshold happen outside ONNX; the encoder and decoder
network passes are pure ONNX. This model is entirely independent of the
project's U-Net, so it never touches the training loop.

The ONNX files live in `sam2/onnx/`; regenerate them with::

    venv/bin/python backend/export_sam_onnx.py --size tiny
"""

import os
import sys
import threading

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a project dependency
    cv2 = None

# Source tree: sam2/onnx next to backend/ (see export_sam_onnx.py).
# Frozen app: sam2/onnx next to this executable -- build_app.sh copies it
# there when present, mirroring how it bundles the standalone predictor.
if getattr(sys, "frozen", False):
    _ONNX_DIR = os.path.join(os.path.dirname(sys.executable), "sam2", "onnx")
else:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ONNX_DIR = os.path.join(_PROJECT_ROOT, "sam2", "onnx")

# Model size -> filename stem of its exported encoder/decoder graphs.
_MODELS = {"tiny": "sam2.1_hiera_tiny", "small": "sam2.1_hiera_small"}

# SAM2 preprocessing constants (ImageNet mean/std, matching SAM2Transforms).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class SamService:
    """Lazy onnxruntime wrapper around the exported SAM2 encoder/decoder."""

    def __init__(self, model_size="tiny"):
        stem = _MODELS[model_size]
        self._encoder_path = os.path.join(_ONNX_DIR, f"{stem}.encoder.onnx")
        self._decoder_path = os.path.join(_ONNX_DIR, f"{stem}.decoder.onnx")
        self._enc = None  # onnxruntime encoder session
        self._dec = None  # onnxruntime decoder session
        self._preprocess = None
        self._res = 1024  # encoder input resolution (read from the graph on load)
        self._mask_res = 256  # decoder mask_input side (read from the graph on load)
        self._feats = None  # cached (image_embed, high_res_feat0, high_res_feat1)
        self._orig_hw = None  # (H, W) of the cached image
        self._image_key = None  # identifies the image whose embeddings are cached
        self._lock = threading.Lock()  # serializes load + encode + decode

    # ---------- availability ----------

    def unavailable_reason(self):
        """None if SAM2 can run here, else a short human-readable reason."""
        if cv2 is None:
            return "OpenCV (cv2) is not installed"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "onnxruntime is not installed"
        for path in (self._encoder_path, self._decoder_path):
            if not os.path.exists(path):
                return ("SAM2 ONNX model missing "
                        f"({os.path.basename(path)}); run "
                        "backend/export_sam_onnx.py")
        return None

    def available(self):
        return self.unavailable_reason() is None

    # ---------- sessions / embeddings ----------

    def _ensure_loaded(self):
        if self._enc is not None:
            return
        import onnxruntime as ort
        from torchvision.transforms import Normalize, Resize, ToTensor

        providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # hide the benign Hiera shape-merge warning
        self._enc = ort.InferenceSession(self._encoder_path, sess_options=opts,
                                         providers=providers)
        self._dec = ort.InferenceSession(self._decoder_path, sess_options=opts,
                                         providers=providers)
        shape = self._enc.get_inputs()[0].shape  # [1, 3, R, R]
        if isinstance(shape[-1], int):
            self._res = shape[-1]
        # Side of the decoder's mask_input grid (its own low-res logits, fed
        # back on refinement); read from the graph so a re-export can change it.
        mask_shape = {i.name: i.shape for i in self._dec.get_inputs()}["mask_input"]
        if isinstance(mask_shape[-1], int):
            self._mask_res = mask_shape[-1]

        to_tensor, resize = ToTensor(), Resize((self._res, self._res))
        normalize = Normalize(_MEAN, _STD)
        # HWC uint8 RGB -> 1x3xRxR float32, matching SAM2Transforms.
        self._preprocess = lambda rgb: (
            normalize(resize(to_tensor(rgb))).unsqueeze(0).numpy())

    def set_image(self, rgb, key):
        """Cache encoder embeddings for an (H, W, 3) uint8 RGB image under `key`.

        Recomputes only when `key` changes, so repeated clicks on the same image
        reuse the encoder pass. `key` should capture everything that changes the
        pixels (project, image name, display bands).
        """
        with self._lock:
            self._ensure_loaded()
            if key is not None and key == self._image_key:
                return
            self._image_key = None  # a failed encode must not leave a stale key
            inp = self._preprocess(np.ascontiguousarray(rgb))
            image_embed, hrf0, hrf1 = self._enc.run(None, {"image": inp})
            self._feats = (image_embed, hrf0, hrf1)
            self._orig_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
            self._image_key = key

    def predict_points(self, points):
        """Object mask (bool H×W) for a list of clicks.

        `points` is [(x, y, label), ...] in image pixels, label 1 = foreground
        ("include this"), 0 = background ("exclude this"). set_image() must have
        been called first. Returns (mask, score).

        Reproduces SAM2's interactive refinement, which is a *chain*: the click
        list is replayed one click at a time and each pass is given the previous
        pass's low-res logits as `mask_input`. Both details matter, and neither
        is optional —

          * **which mask token.** A lone foreground click is ambiguous (roof,
            house, or block?), so the three hypothesis tokens 1-3 compete and
            the best predicted IoU wins. From the second click on, the intent is
            pinned down and the user is refining, so token 0 (SAM2's dedicated
            single-mask output) is used.
          * **feeding the mask back.** Without `mask_input` a background click
            only nudges the mask and need not even exclude the pixel that was
            clicked; with it the mask actually retreats. Measured on a real
            scene: 19,642 px -> 18,585 px with the click still inside the mask,
            versus -> 4,910 px and properly excluded once fed back.

        Replaying the whole chain per call keeps this stateless, so the frontend
        can drop or re-order points and always get the same mask for the same
        click list. The decoder pass is small, and click counts are single
        digits in practice.
        """
        import torch

        if not points:
            raise ValueError("at least one point is required")
        with self._lock:
            if self._feats is None or self._image_key is None:
                raise RuntimeError("SAM2 image embeddings are not set")
            height, width = self._orig_hw
            image_embed, hrf0, hrf1 = self._feats
            prev, score = None, 0.0
            for count in range(1, len(points) + 1):
                chain = points[:count]
                # Clicks in the encoder's input frame (transform_coords).
                coords = np.array(
                    [[[x / width * self._res, y / height * self._res]
                      for x, y, _ in chain]], dtype=np.float32)
                labels = np.array([[int(lab) for _, _, lab in chain]], dtype=np.int64)
                low_res, iou = self._dec.run(None, {
                    "image_embed": image_embed,
                    "high_res_feat0": hrf0,
                    "high_res_feat1": hrf1,
                    "point_coords": coords,
                    "point_labels": labels,
                    "mask_input": (prev if prev is not None else
                                   np.zeros((1, 1, self._mask_res, self._mask_res),
                                            dtype=np.float32)),
                    "has_mask_input": np.array(
                        [0.0 if prev is None else 1.0], dtype=np.float32),
                })
                token = 1 + int(iou[0, 1:].argmax()) if count == 1 else 0
                prev = low_res[:, token:token + 1]
                score = float(iou[0, token])
            # Upscale the final low-res logits to the original size and threshold
            # at 0, exactly like SAM2's postprocess_masks (hole/sprinkle areas 0).
            up = torch.nn.functional.interpolate(
                torch.from_numpy(prev), (height, width),
                mode="bilinear", align_corners=False)
            return up[0, 0].numpy() > 0.0, score


def mask_to_polygons(mask, min_area=16, epsilon_frac=0.004):
    """External contours of a boolean mask as simplified [[x, y], ...] polygons.

    Coordinates are in image pixels. Tiny specks below `min_area` are dropped;
    each contour is simplified with Douglas-Peucker at `epsilon_frac` of its
    perimeter so the vertex list stays small while tracking the edge.
    """
    m = (np.asarray(mask, dtype=np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons
