"""Segmentation nets over the raw input bands, in two bottleneck flavours.

Shared by both: an encoder of residual depthwise-separable blocks with strided
(learned) downsampling, four stages, so the bottleneck sits at 1/16 of the input;
a decoder of transposed-conv upsampling with skip concatenation and channel
attention; and a 1x1 conv head to per-class logits, optionally refined by a
PointRend point head on the most uncertain pixels.

Only the 1/16 stage differs, and it is a per-project setting (see BOTTLENECKS):

    ConvBridgeNetwork      conv blocks       the default
    HybridBridgeNetwork    transformer       kept for higher-resolution imagery

The net emits NUM_CLASSES logits per pixel; softmax over them gives the class
probabilities, and the class index *is* the label value (0 target, 1 background).

Run this module directly for both variants' shapes and parameter breakdown:

    venv/bin/python backend/model.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 96
IN_CHANNELS = 10
NUM_CLASSES = 2
# The point head re-classifies 1/POINT_RATIO of the pixels of a patch. Expressed
# as a ratio rather than a fixed count so it tracks the project's patch size.
POINT_RATIO = 16
POINT_HIDDEN = 64
BASE_CHANNELS = 44
TRANSFORMER_DEPTH = 3
TRANSFORMER_HEADS = 4
STOCHASTIC_DEPTH_PROB = 0.1

# Selectable bottleneck, per project (data_profile.bottleneck). The encoder,
# decoder and PointRend head are identical either way; only the lowest-resolution
# stage differs, along with the channel width each variant is tuned for.
#
#   "conv"         plain conv block (ConvBridgeNetwork), base 64
#   "transformer"  pre-norm transformer encoder (HybridBridgeNetwork), base 44
#
# "conv" is the default on the evidence: measured over 3 seeds on a synthetic
# task with a weak 3-of-10-band signal, it reached mean best IoU 0.535 against
# the transformer's 0.428, and did it with 2.97M parameters against 5.65M. The
# transformer stays available because that measurement is one synthetic tile at
# 10 m/px, where there is plausibly nothing at the scale attention exploits --
# the ordering could well flip on higher-resolution imagery.
BOTTLENECKS = ("conv", "transformer")
DEFAULT_BOTTLENECK = "conv"
# base_channels per variant: replacing the transformer frees ~4M parameters, and
# the conv variant spends part of that on a wider convolutional path instead.
# Channel counts scale roughly quadratically through DoubleConv/Up, so 44 -> 64
# is about 2.1x on every conv stage -- and the net is still 2.97M parameters
# against the transformer variant's 5.65M.
CONV_BASE_CHANNELS = 64
BASE_CHANNELS_BY_BOTTLENECK = {"conv": CONV_BASE_CHANNELS,
                               "transformer": BASE_CHANNELS}
# Stacked DoubleConv blocks at a conv bottleneck. One is the classic U-Net shape
# (its bottleneck is a single two-conv block); 2+ deepens the lowest-resolution
# stage, which is the cheapest place to add depth -- each block is ~536k
# parameters at 64 base channels, so 4 would roughly match the transformer's
# budget if you want the comparison parameter-matched.
BOTTLENECK_BLOCKS = 1

# Weights pretrained on this architecture (kept outside the project), loaded as
# the default initialization when present. They were trained at a different
# patch size, class count and channel count than a given project may use, so the
# load is filtered rather than strict -- see load_pretrained.
if getattr(sys, "frozen", False):
    _ROOT = sys._MEIPASS  # PyInstaller bundle, like main.py
else:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(_ROOT, "pretraining", "pretrained.pth")


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DropPath(nn.Module):
    """Stochastic depth: drops a residual branch for whole samples in training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DoubleConv(nn.Module):
    """Two depthwise-separable convs with a residual (projected) shortcut."""

    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_prob)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = identity + self.drop_path(out)
        out = self.relu(out)
        return out


class TransformerBottleneck(nn.Module):
    """Pre-norm transformer encoder over the flattened bottleneck grid.

    The positional embedding is a learned parameter, so max_tokens fixes the
    largest input the bottleneck accepts; SegmentationModel sizes it from the
    project's patch size.
    """

    def __init__(self, dim, depth, heads, max_tokens=4096, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, norm=nn.LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        n = x.shape[1]
        if n > self.pos_embed.shape[1]:
            raise ValueError(
                f"bottleneck got {n} tokens but the positional embedding holds "
                f"{self.pos_embed.shape[1]} (input larger than the patch size "
                f"the model was built for)"
            )
        x = x + self.pos_embed[:, :n, :]
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation gate on the decoder's output channels."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.gate(self.avg_pool(x))
        return x * scale


class Down(nn.Module):
    """One encoder stage: residual block (the skip) then a strided conv."""

    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop_prob=drop_prob)
        self.pool = nn.Sequential(
            DepthwiseSeparableConv(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class Up(nn.Module):
    """One decoder stage: upsample, concatenate the skip, fuse, gate channels."""

    def __init__(
        self, in_channels, skip_channels, out_channels, reduction=16, drop_prob=0.0
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, drop_prob=drop_prob)
        self.channel_attention = ChannelAttention(out_channels, reduction=reduction)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.channel_attention(x)


class PointRendModule(nn.Module):
    """Per-point MLP that re-classifies the most uncertain coarse logits.

    Kernel-1 Conv1d layers are per-point Linear layers, so cost scales with the
    number of sampled points, not the image size — cheap enough to run on every
    patch during CPU inference. Runs in training too: the refined pixels
    backpropagate into the point head (and its feature inputs), so train.py
    supervises it for free.
    """

    def __init__(
        self,
        fine_channels,
        num_classes,
        hidden_channels=POINT_HIDDEN,
        num_fc_layers=3,
        num_points=4096,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        in_dim = fine_channels + num_classes
        layers = []
        for i in range(num_fc_layers):
            layers.append(
                nn.Conv1d(in_dim if i == 0 else hidden_channels, hidden_channels, kernel_size=1)
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(hidden_channels, num_classes, kernel_size=1))
        self.point_head = nn.Sequential(*layers)

    @staticmethod
    def calculate_uncertainty(logits):
        """Score per pixel; larger = closer to the decision boundary."""
        if logits.shape[1] == 1:
            return -logits.abs()
        top2 = torch.topk(logits, k=2, dim=1)[0]
        return top2[:, 1:2] - top2[:, 0:1]

    @staticmethod
    def point_sample(feature_map, point_coords):
        """Bilinear read of feature_map at (B, K, 2) coords in [0, 1] -> (B, C, K)."""
        grid = 2.0 * point_coords - 1.0
        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False)
        return sampled.squeeze(3)

    def _refine_step(self, logits, prev_logits, fine_features):
        b, c, h, w = logits.shape
        num_points = min(self.num_points, h * w)
        uncertainty = self.calculate_uncertainty(logits).view(b, -1)
        point_indices = torch.topk(uncertainty, k=num_points, dim=1)[1]
        ys = (point_indices // w).float()
        xs = (point_indices % w).float()
        # align_corners=False maps pixel i's center to (i + 0.5) / size.
        point_coords = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h], dim=-1)
        fine_feat = self.point_sample(fine_features, point_coords)
        coarse_feat = self.point_sample(prev_logits, point_coords)
        point_logits = self.point_head(torch.cat([fine_feat, coarse_feat], dim=1))
        logits = logits.reshape(b, c, h * w).clone()
        logits.scatter_(2, point_indices.unsqueeze(1).expand(-1, c, -1), point_logits)
        return logits.view(b, c, h, w)

    def forward(self, coarse_logits, fine_features, target_size=None):
        if target_size is None:
            return self._refine_step(coarse_logits, coarse_logits, fine_features)
        logits = coarse_logits
        while logits.shape[-2] < target_size[-2] or logits.shape[-1] < target_size[-1]:
            prev_logits = logits
            logits = F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)
            logits = self._refine_step(logits, prev_logits, fine_features)
        if logits.shape[-2:] != tuple(target_size):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        return logits


class HybridBridgeNetwork(nn.Module):
    """Residual CNN encoder -> transformer bottleneck -> attention-gated decoder.

    With use_pointrend=False this is exactly the plain encoder/decoder (no extra
    modules, parameters or compute). With True, forward re-classifies the most
    uncertain pixels of the coarse logits with a point head fed by the stride-1
    encoder features — the highest-resolution ones, which still carry the raw
    band edges the decoder has smoothed over — concatenated with the coarse logit.
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS,
        max_input_size=INPUT_SIZE,
        point_rend_num_points=(INPUT_SIZE * INPUT_SIZE) // POINT_RATIO,
        point_rend_hidden_channels=POINT_HIDDEN,
        stochastic_depth_prob=STOCHASTIC_DEPTH_PROB,
        use_pointrend=False,
    ):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        num_blocks = 8
        dpr = [stochastic_depth_prob * i / max(num_blocks - 1, 1) for i in range(num_blocks)]

        # --- ENCODER ---
        self.down1 = Down(in_channels, c1, drop_prob=dpr[0])
        self.down2 = Down(c1, c2, drop_prob=dpr[1])
        self.down3 = Down(c2, c3, drop_prob=dpr[2])
        self.down4 = Down(c3, c4, drop_prob=dpr[3])

        # --- BOTTLENECK ---
        bottleneck_grid = max_input_size // 16
        self.bottleneck = TransformerBottleneck(
            dim=c4,
            depth=transformer_depth,
            heads=transformer_heads,
            max_tokens=bottleneck_grid * bottleneck_grid,
        )

        # --- DECODER ---
        self.up4 = Up(c4, c4, c3, drop_prob=dpr[4])
        self.up3 = Up(c3, c3, c2, drop_prob=dpr[5])
        self.up2 = Up(c2, c2, c1, drop_prob=dpr[6])
        self.up1 = Up(c1, c1, c1, drop_prob=dpr[7])

        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.use_pointrend = use_pointrend
        self.point_rend = (
            PointRendModule(
                fine_channels=c1,
                num_classes=num_classes,
                hidden_channels=point_rend_hidden_channels,
                num_points=point_rend_num_points,
            )
            if use_pointrend
            else None
        )

    def forward(self, x, return_coarse=False):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.bottleneck(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        coarse_logits = self.final_conv(x)  # (B, classes, H, W)
        if self.point_rend is None:
            return (coarse_logits, coarse_logits) if return_coarse else coarse_logits
        refined_logits = self.point_rend(coarse_logits, skip1)
        if return_coarse:
            return refined_logits, coarse_logits
        return refined_logits


class ConvBottleneck(nn.Module):
    """The classic U-Net bottleneck: conv blocks at the lowest resolution.

    Same in/out channel count as TransformerBottleneck, so it drops straight
    into the decoder without touching any Up block. Unlike the transformer it has
    no positional embedding and therefore no dependence on the input size, which
    is what lets its variant accept any patch size that is a multiple of 16, keep
    checkpoints portable across patch sizes, and export to ONNX with dynamic
    spatial axes (see SegmentationModel.variable_input).
    """

    def __init__(self, channels, blocks=BOTTLENECK_BLOCKS, drop_prob=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            DoubleConv(channels, channels, drop_prob=drop_prob)
            for _ in range(blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class ConvBridgeNetwork(nn.Module):
    """HybridBridgeNetwork with ConvBottleneck in place of the transformer.

    The project default (data_profile.bottleneck == "conv"); see BOTTLENECKS for
    the measurements behind that. Everything outside the bottleneck -- the
    encoder, the decoder, the PointRend head -- is the same blocks the
    transformer variant uses, so the bottleneck really is the only difference,
    apart from the wider base_channels that replacing it pays for.

    Keeps HybridBridgeNetwork's forward signature and attribute names (down1..4,
    bottleneck, up4..1, final_conv, point_rend), so train.py's parameter
    grouping, api.py's checkpoint filter and the ONNX export all work against
    either variant without asking which one they have.

    train.BOTTLENECK_LR deliberately does NOT apply here: it exists for the
    attention block, and holding a conv block back at the same rate cost 0.026
    mean best IoU in the A/B. train._bottleneck_lr selects by module type, so
    this bottleneck trains at the full LEARNING_RATE.
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=CONV_BASE_CHANNELS,
        bottleneck_blocks=BOTTLENECK_BLOCKS,
        point_rend_num_points=(INPUT_SIZE * INPUT_SIZE) // POINT_RATIO,
        point_rend_hidden_channels=POINT_HIDDEN,
        stochastic_depth_prob=STOCHASTIC_DEPTH_PROB,
        use_pointrend=False,
    ):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        # One extra entry over HybridBridgeNetwork's eight: the bottleneck is a
        # residual block here too, so it takes a stochastic-depth rate of its own
        # and the schedule stays monotonic across the whole depth of the net.
        num_blocks = 9
        dpr = [stochastic_depth_prob * i / max(num_blocks - 1, 1) for i in range(num_blocks)]

        # --- ENCODER --- (identical to the transformer variant)
        self.down1 = Down(in_channels, c1, drop_prob=dpr[0])
        self.down2 = Down(c1, c2, drop_prob=dpr[1])
        self.down3 = Down(c2, c3, drop_prob=dpr[2])
        self.down4 = Down(c3, c4, drop_prob=dpr[3])

        # --- BOTTLENECK --- (the one architectural difference)
        self.bottleneck = ConvBottleneck(c4, blocks=bottleneck_blocks, drop_prob=dpr[4])

        # --- DECODER --- (identical to the transformer variant)
        self.up4 = Up(c4, c4, c3, drop_prob=dpr[5])
        self.up3 = Up(c3, c3, c2, drop_prob=dpr[6])
        self.up2 = Up(c2, c2, c1, drop_prob=dpr[7])
        self.up1 = Up(c1, c1, c1, drop_prob=dpr[8])

        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.use_pointrend = use_pointrend
        self.point_rend = (
            PointRendModule(
                fine_channels=c1,
                num_classes=num_classes,
                hidden_channels=point_rend_hidden_channels,
                num_points=point_rend_num_points,
            )
            if use_pointrend
            else None
        )

    def forward(self, x, return_coarse=False):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.bottleneck(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        coarse_logits = self.final_conv(x)  # (B, classes, H, W)
        if self.point_rend is None:
            return (coarse_logits, coarse_logits) if return_coarse else coarse_logits
        refined_logits = self.point_rend(coarse_logits, skip1)
        if return_coarse:
            return refined_logits, coarse_logits
        return refined_logits


def _resize_pos_embed(weight, tokens):
    """Resample a (1, N, D) positional embedding to `tokens` positions.

    The embedding is one learned vector per bottleneck cell, so it is really a
    square grid: pretraining at 256px gives a 16x16 grid, a 96px project needs
    6x6. Interpolating the grid carries the learned position information across
    instead of throwing it away, which is what a plain shape filter would do
    (standard practice for transferring ViT position embeddings between
    resolutions). Returns None if either side is not a square grid.
    """
    n, dim = weight.shape[1], weight.shape[2]
    src = int(round(n ** 0.5))
    dst = int(round(tokens ** 0.5))
    if src * src != n or dst * dst != tokens:
        return None
    if src == dst:
        return weight
    grid = weight.reshape(1, src, src, dim).permute(0, 3, 1, 2)  # (1, D, src, src)
    grid = F.interpolate(grid, size=(dst, dst), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, tokens, dim)


def load_pretrained(net, path):
    """Initialize `net` from a pretrained checkpoint; returns tensors loaded.

    Tolerant by design: the checkpoint carries whatever channel count, class
    count and patch size it was pretrained with, and a project rarely matches
    all three. Anything whose name or shape disagrees is left at its fresh
    initialization -- in practice the segmentation head (different class count),
    the point head (it is sized by the class count too) and, for a project with a
    different band count, the first encoder conv. The positional embedding is
    resized rather than dropped. Unreadable checkpoints are ignored, like the
    corrupt-mask fallback in api._read_user_mask: a bad file must not stop the
    app from starting with a freshly initialized model.
    """
    if not path or not os.path.exists(path):
        return 0
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return 0
    state = state.get("state_dict", state) if isinstance(state, dict) else {}
    # Checkpoints written by a training wrapper keep the net under `model.`.
    state = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v
             for k, v in state.items()}
    target = net.state_dict()
    if "bottleneck.pos_embed" in state and "bottleneck.pos_embed" in target:
        resized = _resize_pos_embed(
            state["bottleneck.pos_embed"], target["bottleneck.pos_embed"].shape[1]
        )
        if resized is None:
            del state["bottleneck.pos_embed"]
        else:
            state["bottleneck.pos_embed"] = resized
    usable = {
        k: v
        for k, v in state.items()
        if k in target and v.shape == target[k].shape
    }
    net.load_state_dict(usable, strict=False)
    return len(usable)


class SegmentationModel:
    """Wraps the network with patch-tiled inference and (de)serialization.

    in_channels / patch_size / bottleneck come from the project's data_profile.
    A trained checkpoint is only portable between projects that agree on all
    three: the bottleneck decides the architecture, and under "transformer" the
    patch size additionally sizes the positional embedding. The pretrained
    weights are adapted across what can be adapted by load_pretrained.
    """

    def __init__(
        self,
        device=None,
        weights=DEFAULT_WEIGHTS,
        in_channels=IN_CHANNELS,
        patch_size=INPUT_SIZE,
        use_pointrend=False,
        bottleneck=DEFAULT_BOTTLENECK,
    ):
        if patch_size % 16 != 0:
            raise ValueError(
                f"input_patch_size must be a multiple of 16 "
                f"(encoder downsamples 16x), got {patch_size}"
            )
        if bottleneck not in BOTTLENECKS:
            raise ValueError(
                f"bottleneck must be one of {BOTTLENECKS}, got {bottleneck!r}"
            )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.use_pointrend = use_pointrend
        self.bottleneck = bottleneck
        base_channels = BASE_CHANNELS_BY_BOTTLENECK[bottleneck]
        points = max(1, (patch_size * patch_size) // POINT_RATIO)
        if bottleneck == "conv":
            net = ConvBridgeNetwork(
                in_channels=in_channels,
                num_classes=NUM_CLASSES,
                base_channels=base_channels,
                point_rend_num_points=points,
                use_pointrend=use_pointrend,
            )
        else:
            net = HybridBridgeNetwork(
                in_channels=in_channels,
                num_classes=NUM_CLASSES,
                base_channels=base_channels,
                max_input_size=patch_size,
                point_rend_num_points=points,
                use_pointrend=use_pointrend,
            )
        self.net = net.to(self.device)
        self.pretrained_tensors = load_pretrained(self.net, weights)

    @classmethod
    def for_project(cls, project, **overrides):
        """Build the model a project's data_profile describes.

        The one place that knows which profile fields define the network, so
        adding a fifth knob is one edit here rather than one per call site.
        Duck-typed on the four properties rather than importing Project, which
        would be a cycle (project.py imports this module for BOTTLENECKS).
        """
        return cls(in_channels=project.input_channels,
                   patch_size=project.patch_size,
                   use_pointrend=project.use_pointrend,
                   bottleneck=project.bottleneck,
                   **overrides)

    @property
    def profile(self):
        """The kwargs that define this network, for rebuilding it identically.

        Used where the profile has to come from the live model rather than from a
        project -- `reset_model` runs on a standalone image too, where there is no
        project to read.
        """
        return {"in_channels": self.in_channels,
                "patch_size": self.patch_size,
                "use_pointrend": self.use_pointrend,
                "bottleneck": self.bottleneck}

    @property
    def variable_input(self):
        """Whether the net accepts input sizes other than the one it was built for.

        The transformer bottleneck's positional embedding is a learned parameter
        sized for exactly one token grid, so that variant is pinned to its patch
        size. The conv bottleneck has no size-dependent parameter and runs at any
        multiple of 16 (the encoder's total stride).

        Read by api._write_model_onnx to decide whether the exported graph may
        declare its spatial axes dynamic.
        """
        return self.bottleneck == "conv"

    def predict_image(self, image):
        """Full-image class map via overlapping tiles with logit averaging.

        image: data.SentinelImage. Returns (class_map (H, W) uint8, probs (C, H, W)).
        class_map is in label space (= model class space): 0 target, 1 background,
        255 where the image has no valid data.
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
        logit_map = blend_tiles((image.height, image.width), tiles)  # (C, H, W)
        probs = torch.softmax(torch.from_numpy(logit_map), dim=0).numpy()
        # The row index is the class label, so probs keeps its (C, H, W) contract
        # and argmax lands directly in label space.
        class_map = probs.argmax(axis=0).astype(np.uint8)
        class_map[~image.valid_mask] = UNLABELED
        return class_map, probs

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


def _compare_variants():
    """Side-by-side shapes and parameter split for the two bottlenecks.

    Handy when tuning either one: the per-module columns show where the budget
    goes, and the totals show what swapping the bottleneck costs or frees.
    """
    def size_of(module):
        return sum(p.numel() for p in module.parameters())

    built = {name: SegmentationModel(in_channels=IN_CHANNELS, patch_size=INPUT_SIZE,
                                     bottleneck=name, use_pointrend=True,
                                     weights=None).net
             for name in BOTTLENECKS}
    dummy = torch.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    for name, variant in built.items():
        variant.eval()
        with torch.no_grad():
            refined, coarse = variant(dummy, return_coarse=True)
        print(f"{name:12s} in {tuple(dummy.shape)} -> coarse {tuple(coarse.shape)}"
              f" -> refined {tuple(refined.shape)}")

    conv, hybrid = built["conv"], built["transformer"]
    print(f"\n{'module':<14}{f'conv (base {CONV_BASE_CHANNELS})':>20}"
          f"{f'transformer (base {BASE_CHANNELS})':>24}")
    for name, module in conv.named_children():
        twin = getattr(hybrid, name, None)
        twin_size = f"{size_of(twin):,}" if twin is not None else "-"
        print(f"  {name:<12}{size_of(module):>20,}{twin_size:>24}")
    print(f"  {'TOTAL':<12}{size_of(conv):>20,}{size_of(hybrid):>24,}")

    # Only the conv variant is size-agnostic: no positional embedding, so no
    # largest-input constraint (see SegmentationModel.variable_input).
    for side in (64, 128, 256):
        with torch.no_grad():
            logits = conv(torch.randn(1, IN_CHANNELS, side, side))
        print(f"conv at {side:>4}px  {tuple(logits.shape)}")


if __name__ == "__main__":
    _compare_variants()
