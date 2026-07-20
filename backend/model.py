"""U-Net with a pre-trained EfficientNet-B0 encoder over the raw input bands."""

import os

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

INPUT_SIZE = 96
IN_CHANNELS = 10
OUT_LOGITS = 1
POINT_RATIO = 16
POINT_HIDDEN = 64
# Weights produced by ../pretraining/pretrain.py (kept outside the project);
# loaded (minus the segmentation head, whose class count differs) as the
# default initialization when present.
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    "pretraining",
    "pretrained.pth",
)


class PointHead(torch.nn.Module):
    """Per-point MLP over (B, C, N) feature stacks: (B, classes, N) logits.

    Kernel-1 Conv1d layers are per-point Linear layers, so cost scales with the
    number of sampled points, not the image size — a few thousand parameters,
    cheap enough to run on every patch during CPU inference.
    """

    def __init__(self, in_channels, classes=OUT_LOGITS, hidden=POINT_HIDDEN):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, hidden, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(hidden, hidden, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(hidden, classes, 1),
        )

    def forward(self, point_features):
        return self.mlp(point_features)


class BandUnet(torch.nn.Module):
    """The U-Net, optionally wrapped with a PointRend-style refinement head.

    With use_pointrend=False this is exactly the plain U-Net (no extra modules,
    parameters or compute). With True, forward re-classifies the most uncertain
    pixels of the coarse logits with a PointHead fed by the stride-2 encoder
    features — the highest-resolution ones, which still carry the raw band
    edges the decoder has smoothed over — concatenated with the coarse logit.
    """

    def __init__(
        self, in_channels=IN_CHANNELS, classes=OUT_LOGITS, use_pointrend=False
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            decoder_interpolation="bilinear",
        )
        self.use_pointrend = use_pointrend
        self.point_head = (
            PointHead(self.unet.encoder.out_channels[1] + classes)
            if use_pointrend
            else None
        )

    def forward(self, x):
        if not self.use_pointrend:
            return self.unet(x)  # (B, classes, H, W)
        # smp's forward inlined so the encoder features stay available.
        features = self.unet.encoder(x)
        logits = self.unet.segmentation_head(self.unet.decoder(features))
        return self._refine(logits, features[1])

    @staticmethod
    def uncertainty_sampling(logits, k):
        """Flat indices (B, k) of the k pixels closest to the decision boundary.

        Single-logit binary output: sigmoid(0) = 0.5, so |logit| is the
        distance from P(target) = 0.5 and the smallest-|logit| pixels win.
        """
        flat = logits.detach().flatten(1)  # (B, H*W); classes == 1
        return flat.abs().topk(min(k, flat.shape[1]), dim=1, largest=False).indices

    def _refine(self, logits, fine):
        """Replace the most uncertain coarse logits with point-head logits.

        Runs in training too: the refined pixels backpropagate into the point
        head (and its feature inputs), so train.py supervises it for free.
        """
        B, C, H, W = logits.shape
        idx = self.uncertainty_sampling(logits, max(1, (H * W) // POINT_RATIO))
        # Flat indices -> pixel-center coordinates in grid_sample's [-1, 1]
        # space (align_corners=False maps pixel i's center to (i + 0.5) / size).
        xs = (idx % W).float()
        ys = (idx // W).float()
        grid = torch.stack(
            [(xs + 0.5) * (2.0 / W) - 1.0, (ys + 0.5) * (2.0 / H) - 1.0], dim=-1
        ).unsqueeze(
            2
        )  # (B, K, 1, 2)
        fine_feats = F.grid_sample(
            fine, grid.to(fine.dtype), align_corners=False
        ).squeeze(
            -1
        )  # (B, C_fine, K), bilinear at the stride-2 map
        # The points are exact pixel centers of the logit map itself, so a
        # gather reads the coarse logits there without interpolation.
        index = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, K)
        coarse = logits.flatten(2).gather(2, index)
        # Explicit casts so enabling AMP later cannot break this: autocast keeps
        # grid_sample in fp32 while the conv logits go half, and cat/scatter
        # require exact dtype matches. No-ops in the fp32 path used today.
        point_logits = self.point_head(
            torch.cat([fine_feats.to(coarse.dtype), coarse], dim=1)
        )
        return (
            logits.flatten(2)
            .scatter(2, index, point_logits.to(logits.dtype))
            .view(B, C, H, W)
        )


class SegmentationModel:
    """Wraps the U-Net with patch-tiled inference and (de)serialization.

    in_channels / patch_size come from the project's data_profile; the
    Sentinel-2 pretrained weights only apply to the 10-band configuration,
    other channel counts start from the ImageNet encoder initialization.
    """

    def __init__(
        self,
        device=None,
        weights=DEFAULT_WEIGHTS,
        in_channels=IN_CHANNELS,
        patch_size=INPUT_SIZE,
        use_pointrend=False,
    ):
        if patch_size % 32 != 0:
            raise ValueError(
                f"input_patch_size must be a multiple of 32 "
                f"(encoder downsamples 32x), got {patch_size}"
            )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_pointrend = use_pointrend
        self.net = BandUnet(in_channels, use_pointrend=use_pointrend).to(self.device)
        if in_channels == IN_CHANNELS and weights and os.path.exists(weights):
            state = torch.load(weights, map_location=self.device)
            model_state = self.net.state_dict()
            state = {
                k: v
                for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            self.net.load_state_dict(state, strict=False)

    def predict_image(self, image):
        """Full-image class map via overlapping tiles with logit averaging.

        image: data.SentinelImage. Returns (class_map (H, W) uint8, probs (C, H, W)).
        class_map is in label space: 0 target, 1 background, 255 where the
        image has no valid data.
        """
        from data import UNLABELED, blend_tiles

        self.net.eval()
        corners = image.patch_grid()
        tiles = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(corners), batch_size):
                chunk = corners[i : i + batch_size]
                x = torch.from_numpy(
                    np.stack([image.patch(y, c) for y, c in chunk])
                ).to(self.device)
                logits = self.net(x).cpu().numpy()
                tiles.extend(zip(chunk, logits))
        logit_map = blend_tiles((image.height, image.width), tiles)  # (1, H, W)
        p_target = torch.sigmoid(torch.from_numpy(logit_map[0])).numpy()
        # Row index stays the class label, so probs keeps its (C, H, W) contract.
        probs = np.stack([p_target, 1.0 - p_target])
        class_map = (p_target <= 0.5).astype(np.uint8)  # 0 target, 1 background
        class_map[~image.valid_mask] = UNLABELED
        return class_map, probs

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
