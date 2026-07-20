"""Export Efficient-SAM2 to ONNX (image encoder + prompt/mask decoder).

Run once to produce the two ONNX graphs the app uses at runtime (see
`sam_service.py`); PyTorch/hydra are only needed here, not when the app runs.

    venv/bin/python backend/export_sam_onnx.py [--size tiny|small]

The image predictor is split the same way the PyTorch predictor is used:

  * encoder: preprocessed 1x3x1024x1024 image -> image_embed + 2 high-res feats
  * decoder: those feats + clicks (point_coords/labels in the 1024 frame; label
             1 = foreground, 0 = background, any number of points)
             -> low-res mask logits (1x4x256x256) + IoU predictions (1x4)

Both decoder regimes are exported: mask token 0 is SAM2's single refinement
mask (used once the click has been disambiguated by further points), tokens 1-3
are the ambiguity hypotheses for a lone first click. See `PromptMaskDecoder`.

Outputs land in `sam2/onnx/`. The script verifies the ONNX pipeline matches the
PyTorch predictor before declaring success.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAM_DIR = os.path.join(_PROJECT_ROOT, "sam2")
_REPO_DIR = os.path.join(_SAM_DIR, "Efficient-SAM2")
_ONNX_DIR = os.path.join(_SAM_DIR, "onnx")

_MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
}

# Spatial sizes of the three backbone feature maps (from SAM2ImagePredictor).
_BB_FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]
_OPSET = 17


class ImageEncoder(nn.Module):
    """Reproduces SAM2ImagePredictor.set_image's embedding computation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):  # image: 1x3xRxR, already resized + normalized
        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *size)
            for feat, size in zip(vision_feats[::-1], _BB_FEAT_SIZES[::-1])
        ][::-1]
        # feats = [high_res_feat0, high_res_feat1, image_embed]
        return feats[-1], feats[0], feats[1]


class PromptMaskDecoder(nn.Module):
    """Reproduces SAM2ImagePredictor._predict for a points-only prompt.

    Calls `predict_masks` directly rather than the decoder's `forward`, so all
    **four** mask tokens survive into the graph. `forward` would slice them
    (`multimask_output=True` -> tokens 1:4, `False` -> token 0:1), and the
    caller needs both regimes:

      * token 0    -- the dedicated single-mask token SAM2 uses for *refinement*
                      clicks (its `multimask_output=False` path; the decoder's
                      `dynamic_multimask_via_stability` is False here, so that
                      path is exactly this slice).
      * tokens 1-3 -- the ambiguity hypotheses used for a *first* click, where
                      one point is ambiguous and the best IoU wins.

    Exporting both avoids a bool input branching the graph (ONNX handles that
    badly) and keeps a single decoder file. `sam_service.py` does the slicing.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_embed, high_res_feat0, high_res_feat1, point_coords,
                point_labels, mask_input, has_mask_input):
        prompt_encoder = self.model.sam_prompt_encoder
        sparse, _ = prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        # Dense prompt = the previous mask's logits, which is how SAM2 refines
        # across clicks (without it a background click barely moves the mask and
        # need not even exclude the clicked pixel). `masks=None` vs a real mask
        # is a Python branch in the prompt encoder, so instead of baking one in,
        # blend both with a 0/1 flag — the standard SAM ONNX trick, keeping a
        # single graph for the first click (0) and refinements (1).
        embed_h, embed_w = prompt_encoder.image_embedding_size
        dense_mask = prompt_encoder._embed_masks(mask_input)
        dense_none = prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, embed_h, embed_w)
        dense = has_mask_input * dense_mask + (1.0 - has_mask_input) * dense_none
        low_res_masks, iou_pred, _, _ = self.model.sam_mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[high_res_feat0, high_res_feat1],
        )
        return low_res_masks, iou_pred


def _build_model(size):
    sys.path.insert(0, _REPO_DIR)
    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    from sam2.build_sam import build_sam2

    config, ckpt_name = _MODELS[size]
    ckpt = os.path.join(_SAM_DIR, ckpt_name)
    # apply_postprocessing=False keeps the decoder graph clean; it only affects
    # the single-mask path, which we never use (we take argmax over the 3 masks).
    model = build_sam2(config, ckpt, device="cpu", apply_postprocessing=False)
    model.eval()
    return model


def export(size="tiny"):
    os.makedirs(_ONNX_DIR, exist_ok=True)
    model = _build_model(size)
    res = model.image_size
    print(f"model.image_size = {res}")

    enc_path = os.path.join(_ONNX_DIR, f"sam2.1_hiera_{size}.encoder.onnx")
    dec_path = os.path.join(_ONNX_DIR, f"sam2.1_hiera_{size}.decoder.onnx")

    # ---- encoder ----
    encoder = ImageEncoder(model)
    image = torch.randn(1, 3, res, res)
    with torch.no_grad():
        image_embed, hrf0, hrf1 = encoder(image)
    print(f"encoder outputs: image_embed{tuple(image_embed.shape)} "
          f"hrf0{tuple(hrf0.shape)} hrf1{tuple(hrf1.shape)}")
    torch.onnx.export(
        encoder, (image,), enc_path,
        input_names=["image"],
        output_names=["image_embed", "high_res_feat0", "high_res_feat1"],
        opset_version=_OPSET,
        dynamo=False,
    )
    print(f"wrote {enc_path}")

    # ---- decoder ----
    decoder = PromptMaskDecoder(model)
    # Two points (one fg, one bg) so tracing exercises the dynamic num_points
    # axis and the background-label branch of the prompt encoder.
    point_coords = torch.tensor(                                              # 1x2x2
        [[[res / 2, res / 2], [res / 4, res / 4]]], dtype=torch.float32)
    point_labels = torch.tensor([[1, 0]], dtype=torch.int64)                  # 1x2
    # mask_input is the decoder's own low-res logits fed back on refinement:
    # 4x the prompt encoder's embedding grid (64 -> 256), per SAM2Transforms.
    mask_res = 4 * model.sam_prompt_encoder.image_embedding_size[0]
    mask_input = torch.zeros(1, 1, mask_res, mask_res, dtype=torch.float32)
    has_mask_input = torch.ones(1, dtype=torch.float32)
    torch.onnx.export(
        decoder,
        (image_embed, hrf0, hrf1, point_coords, point_labels, mask_input, has_mask_input),
        dec_path,
        input_names=["image_embed", "high_res_feat0", "high_res_feat1",
                     "point_coords", "point_labels", "mask_input", "has_mask_input"],
        output_names=["low_res_masks", "iou_predictions"],
        opset_version=_OPSET,
        dynamic_axes={"point_coords": {1: "num_points"},
                      "point_labels": {1: "num_points"}},
        dynamo=False,
    )
    print(f"wrote {dec_path}")

    _verify(model, enc_path, dec_path, res)


def _verify(model, enc_path, dec_path, res):
    """Compare the ONNX pipeline against the PyTorch predictor on a random image."""
    import onnxruntime as ort
    from torchvision.transforms import Normalize, Resize, ToTensor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    rng = np.random.default_rng(0)
    H, W = 341, 512
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    px, py = W // 2, H // 2

    # PyTorch reference
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(rgb)
    masks_t, scores_t, low_t = predictor.predict(
        point_coords=np.array([[px, py]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    best_t = int(scores_t.argmax())

    # ONNX pipeline
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    to_tensor, resize = ToTensor(), Resize((res, res))
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    inp = norm(resize(to_tensor(rgb))).unsqueeze(0).numpy()
    ie, h0, h1 = enc.run(None, {"image": inp})

    # embedding parity
    embed_t = predictor._features["image_embed"].numpy()
    print(f"encoder image_embed max|Δ| = {np.abs(ie - embed_t).max():.3e}")

    mask_res = 4 * model.sam_prompt_encoder.image_embedding_size[0]

    def run_onnx(points, labels, prev=None):
        pc = np.array([[[x / W * res, y / H * res] for x, y in points]], dtype=np.float32)
        pl = np.array([labels], dtype=np.int64)
        mi = (prev if prev is not None
              else np.zeros((1, 1, mask_res, mask_res), dtype=np.float32))
        hm = np.array([1.0 if prev is not None else 0.0], dtype=np.float32)
        return dec.run(None, {"image_embed": ie, "high_res_feat0": h0,
                              "high_res_feat1": h1, "point_coords": pc,
                              "point_labels": pl, "mask_input": mi,
                              "has_mask_input": hm})

    def upscale(low_res, token):
        up = torch.nn.functional.interpolate(
            torch.from_numpy(low_res[:, token:token + 1]), (H, W),
            mode="bilinear", align_corners=False)
        return up[0, 0].numpy() > 0.0

    def mask_iou(a, b):
        union = np.logical_or(a, b).sum()
        return np.logical_and(a, b).sum() / union if union else 1.0

    # --- first click: ONNX tokens 1-3 == torch multimask_output=True ---
    lr, iou = run_onnx([(px, py)], [1])
    assert lr.shape[1] == 4, f"expected 4 mask tokens, got {lr.shape[1]}"
    best_o = 1 + int(iou[0, 1:].argmax())  # tokens 1-3; torch index is best_o - 1
    mask_o = upscale(lr, best_o)
    mask_t = masks_t[best_t] >= 0.5
    iou_multi = mask_iou(mask_o, mask_t)
    print(f"iou scores  torch={np.round(scores_t, 4)}  onnx[1:]={np.round(iou[0, 1:], 4)}")
    print(f"multimask  mask IoU (onnx vs torch) = {iou_multi:.4f}")

    # --- refinement: exactly the flow the app uses -- token 0, plus the
    # previous mask's low-res logits fed back as mask_input. Without that
    # feedback a background click barely moves the mask and need not even
    # exclude the clicked pixel, so this asserts both parity and exclusion.
    ys, xs = np.nonzero(mask_o)
    assert len(xs), "reference mask is empty; cannot place a negative point"
    mid = len(xs) // 2
    nx, ny = int(xs[mid]), int(ys[mid])
    masks_r, scores_r, _ = predictor.predict(
        point_coords=np.array([[px, py], [nx, ny]], dtype=np.float32),
        point_labels=np.array([1, 0], dtype=np.int32),
        multimask_output=False,
        mask_input=low_t[best_t:best_t + 1],
    )
    lr_r, iou_r = run_onnx([(px, py), (nx, ny)], [1, 0],
                           prev=lr[:, best_o:best_o + 1])
    mask_or = upscale(lr_r, 0)
    mask_tr = masks_r[0] >= 0.5
    iou_refine = mask_iou(mask_or, mask_tr)
    print(f"refine (+1 -1) mask IoU (onnx vs torch) = {iou_refine:.4f}  "
          f"torch score={scores_r[0]:.4f} onnx score={iou_r[0, 0]:.4f}")
    print(f"refined size onnx={mask_or.sum():,} px  torch={mask_tr.sum():,} px "
          f"(from {mask_o.sum():,} px)")
    print(f"negative point excluded: onnx={not bool(mask_or[ny, nx])} "
          f"torch={not bool(mask_tr[ny, nx])}")

    assert np.abs(ie - embed_t).max() < 1e-3, "encoder embeddings diverge"
    assert iou_multi > 0.99, "multimask decoder mask diverges from PyTorch"
    assert iou_refine > 0.99, "refinement (token 0 + mask_input) diverges from PyTorch"
    assert not mask_or[ny, nx], "background click did not exclude its own pixel"
    print("PARITY OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(_MODELS), default="tiny")
    export(ap.parse_args().size)
