"""Semi-supervised training: supervised CE + Lovasz + Dice, pseudo-labeling, consistency.

FixMatch-style scheme on patches (size from the image's data profile):
- a *weak* view (geometric augs, applied jointly to image and label mask) is
  forwarded without gradients to produce targets,
- a *strong* view (photometric noise on top of the weak view, so it stays
  pixel-aligned) is forwarded with gradients and receives all three losses.

User-labeled pixels are excluded from the pseudo-label and consistency masks,
so manual ground truth always overrides model beliefs.

The net emits one logit per class per pixel and the class index is the label
value, so every target here is a plain (B, H, W) label map with UNLABELED as the
ignore index. The supervised term pairs a class-weighted per-pixel cross-entropy
with two overlap surrogates -- Lovasz, which optimizes IoU directly, and Dice at
half weight; the pixel-wise unsupervised losses keep cross-entropy alone.
"""

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from data import UNLABELED
from model import TransformerBottleneck

EXCLUDED = 254
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
BOTTLENECK_LR = 2e-4
CONFIDENCE_THRESHOLD = 0.92

LAMBDA_LOVASZ = 1.0
LAMBDA_DICE = 0.5

CLASS_WEIGHTS = (2.0, 1.0)
LAMBDA_PSEUDO = 0.5
LAMBDA_CONSISTENCY = 1.0

TARGET_IOU_STABILITY = 0.70  # live target IoU the model must reach...
STABILITY_PATIENCE = 3  # ...and hold for this many consecutive epochs

RAMP_FRACTION = 0.6
RAMP_MIN_EPOCHS = 5

weak_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

strong_transform = A.Compose(
    [
        A.GaussNoise(std_range=(0.01, 0.05), p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.ChannelDropout(channel_drop_range=(1, 2), fill=0.0, p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(4, 24),
            hole_width_range=(4, 24),
            fill=0.0,
            p=0.5,
        ),
    ]
)

# ---------- multi-patch augmentations ----------
TARGET = 0
P_MOSAIC = 0.2
P_CUTMIX = 0.2
P_COPY_PASTE = 0.2
CUTMIX_AREA = (0.05, 0.35)
CUTMIX_ASPECT = (0.5, 2.0)
DONOR_ATTEMPTS = 8


def _bottleneck_lr(bottleneck):
    """BOTTLENECK_LR for an attention bottleneck, LEARNING_RATE for a conv one.

    Selected by module type rather than by the project's config string, so the
    rate follows the architecture that is actually in the net and cannot drift
    out of step with it.

    Only the transformer needs holding back: it is the sole attention block and
    carries ~79% of that variant's parameters, so it is what destabilizes first.
    A conv bottleneck is the same kind of block as the encoder around it and
    wants the same rate -- measured, holding it at the attention rate cost 0.026
    mean best IoU over three seeds.
    """
    return BOTTLENECK_LR if isinstance(bottleneck, TransformerBottleneck) else LEARNING_RATE


def _sample_patch(image, labels, rng, require=None):
    """A random (patch HWC, labels) pair from anywhere on the image.

    Sampled off-grid (any valid top-left), so donors are not restricted to the
    same half-overlapping corners the training batch walks. `require` is an
    optional predicate on the label window; when it never holds within
    DONOR_ATTEMPTS tries this returns None and the caller leaves the patch
    alone — a project with no target labels must not spin here.
    """
    size = image.patch_size
    for _ in range(DONOR_ATTEMPTS if require else 1):
        y = int(rng.integers(0, image.height - size + 1))
        x = int(rng.integers(0, image.width - size + 1))
        mask = labels[y : y + size, x : x + size]
        if require is None or require(mask):
            return np.transpose(image.patch(y, x), (1, 2, 0)).copy(), mask.copy()
    return None


def _has_labels(mask):
    return bool(np.any((mask != UNLABELED) & (mask != EXCLUDED)))


def _has_target(mask):
    return bool(np.any(mask == TARGET))


def mosaic(img, mask, image, labels, rng, require=None):
    """Rebuild the patch from four patches meeting at a random split point.

    The patch keeps its own top-left quadrant; the other three come from fresh
    donors at the same coordinates. Every tile is real imagery at its original
    scale (no resizing), so only the seams are synthetic.
    """
    h, w = mask.shape
    # Keep the split off the edges so all four tiles actually contribute.
    cy = int(rng.integers(h // 4, h - h // 4 + 1))
    cx = int(rng.integers(w // 4, w - w // 4 + 1))
    out_img, out_mask = img.copy(), mask.copy()
    for ys, xs in (((0, cy), (cx, w)), ((cy, h), (0, cx)), ((cy, h), (cx, w))):
        donor = _sample_patch(image, labels, rng, require)
        if donor is None:
            continue
        d_img, d_mask = donor
        rows, cols = slice(*ys), slice(*xs)
        out_img[rows, cols] = d_img[rows, cols]
        out_mask[rows, cols] = d_mask[rows, cols]
    return out_img, out_mask


def cutmix(img, mask, image, labels, rng, require=None):
    """Replace one random box with the same box from another patch."""
    donor = _sample_patch(image, labels, rng, require)
    if donor is None:
        return img, mask
    d_img, d_mask = donor
    h, w = mask.shape
    area = float(rng.uniform(*CUTMIX_AREA)) * h * w
    aspect = float(rng.uniform(*CUTMIX_ASPECT))
    bh = min(h, max(1, int(round((area / aspect) ** 0.5))))
    bw = min(w, max(1, int(round((area * aspect) ** 0.5))))
    y = int(rng.integers(0, h - bh + 1))
    x = int(rng.integers(0, w - bw + 1))
    out_img, out_mask = img.copy(), mask.copy()
    out_img[y : y + bh, x : x + bw] = d_img[y : y + bh, x : x + bw]
    out_mask[y : y + bh, x : x + bw] = d_mask[y : y + bh, x : x + bw]
    return out_img, out_mask


def copy_paste(img, mask, image, labels, rng):
    """Paste another patch's target pixels onto this one, labels included.

    Only pixels the user labeled as target are copied, so the pasted region
    arrives with ground truth rather than a model guess. Flips vary the shape
    so a donor is not reproduced pixel-for-pixel. Pixels the receiving patch
    marks EXCLUDED (source nodata) are left alone: painting imagery into a
    nodata hole would contradict the image's own valid_mask.
    """
    donor = _sample_patch(image, labels, rng, require=_has_target)
    if donor is None:
        return img, mask  # nothing labeled target anywhere on this image
    d_img, d_mask = donor
    take = d_mask == TARGET
    if rng.random() < 0.5:
        d_img, take = d_img[:, ::-1], take[:, ::-1]
    if rng.random() < 0.5:
        d_img, take = d_img[::-1], take[::-1]
    take = take & (mask != EXCLUDED)
    if not take.any():
        return img, mask
    out_img, out_mask = img.copy(), mask.copy()
    out_img[take] = d_img[take]
    out_mask[take] = TARGET
    return out_img, out_mask


def compose_patch(img, mask, image, labels, rng, prefer_labeled=False):
    """Apply the multi-patch augmentations, each with its own probability.

    prefer_labeled makes mosaic and cutmix look for donors that carry user
    annotation. The warm-up passes it: that phase deliberately trains only on
    labeled patches, and splicing in blank ones would quietly undo the saving
    (the supervised losses are masked to labeled pixels, so blank donors cost a
    forward pass and teach nothing).
    """
    require = _has_labels if prefer_labeled else None
    if rng.random() < P_MOSAIC:
        img, mask = mosaic(img, mask, image, labels, rng, require)
    if rng.random() < P_CUTMIX:
        img, mask = cutmix(img, mask, image, labels, rng, require)
    if rng.random() < P_COPY_PASTE:
        img, mask = copy_paste(img, mask, image, labels, rng)
    return img, mask


def _make_batches(image, labels, rng, skip_unlabeled=False):
    """Yield (weak, strong, label) tensors over a shuffled pass of the patch grid.

    skip_unlabeled drops patches carrying no user annotation at all. Both
    supervised losses are identically zero on those, so during the warm-up they
    buy nothing while still driving an optimizer step. Only the warm-up passes
    it: once the unsupervised losses are live, fully-unlabeled patches are the
    bulk of what pseudo-labeling and consistency learn from.

    The test reads the mask before the patch load and the augmentations. The
    geometric ones cannot change which pixels are labeled, but copy_paste can
    (it brings target labels with it), so the order matters: a patch with no
    annotation of its own is dropped during the warm-up even though pasting
    could have given it some. That keeps the warm-up's saving intact and its
    selection honest — a patch earns its place by what the user drew on it.
    """
    corners = image.patch_grid()
    rng.shuffle(corners)
    for i in range(0, len(corners), BATCH_SIZE):
        weak_imgs, strong_imgs, masks = [], [], []
        for y, x in corners[i : i + BATCH_SIZE]:
            mask = labels[y : y + image.patch_size, x : x + image.patch_size]
            if skip_unlabeled and not np.any((mask != UNLABELED) & (mask != EXCLUDED)):
                continue
            img_hwc = np.transpose(
                image.patch(y, x), (1, 2, 0)
            )

            img_hwc, mask = compose_patch(
                img_hwc, mask, image, labels, rng, prefer_labeled=skip_unlabeled
            )
            out = weak_transform(image=img_hwc, mask=mask)
            weak = out["image"]
            strong = strong_transform(image=weak)["image"]
            weak_imgs.append(np.transpose(weak, (2, 0, 1)))
            strong_imgs.append(np.transpose(strong, (2, 0, 1)))
            masks.append(out["mask"])
        if not weak_imgs:  # every patch in this slice was skipped
            continue
        yield (
            torch.from_numpy(np.stack(weak_imgs)),
            torch.from_numpy(np.stack(strong_imgs)),
            torch.from_numpy(np.stack(masks).astype(np.int64)),
        )


def train(model, samples, epochs, progress=None, stop_event=None):
    """Run the hybrid loop for `epochs` epochs over every labeled image.

    model: model.SegmentationModel;
    samples: sequence of zero-arg callables, each returning one
    (data.SentinelImage, labels (H, W) uint8) pair. They are called once per
    image per epoch and the result is dropped afterwards, so only a single
    image is resident at a time no matter how large the project is. Labels are
    in label space (= model class space): 0 target, 1 background, 255 unlabeled;
    progress: callback(dict) after each epoch; stop_event: threading.Event.
    Returns the number of epochs actually completed, which is `epochs` unless
    stopped early.

    An epoch is one shuffled pass over the patch grids of all samples, so the
    reported losses average over the whole labeled set.

    The run opens purely supervised: the unsupervised losses are gated off and
    fully-unlabeled patches are skipped, so the model first fits the explicit
    user labels. The gate opens on measured quality rather than on an epoch
    count -- the live target IoU must reach TARGET_IOU_STABILITY and hold it for
    STABILITY_PATIENCE consecutive epochs -- because training on the model's own
    predictions is only worth doing once those predictions are, and a model that
    is merely passing through a good epoch will bake its noise in. Afterwards
    pseudo-labeling and consistency fade in over RAMP_FRACTION of the epochs
    that remain, so the fade stretches with the run instead of always taking a
    fixed handful of epochs.

    The gate latches: once open it stays open for the rest of the run. A later
    IoU dip must not flip fully-unlabeled patches back out of the batches or
    rewind the ramp, which would leave the loss composition oscillating.

    Because the gate is measured, not scheduled, it is scoped to this run: a
    resumed run re-earns it (STABILITY_PATIENCE epochs, since the model is
    already good) and re-ramps from there rather than resuming at full weight.

    `epochs` is run as asked, with no minimum imposed: the gate already
    withholds the unsupervised phase from a model that has not proved itself, so
    a run too short to reach it simply stays supervised throughout instead of
    needing to be lengthened into safety.
    """
    if not samples:
        raise ValueError("No labeled images to train on")
    epochs = max(1, int(epochs))  # a run of zero epochs is a no-op, not a request
    net = model.net
    device = model.device

    # reduction="none" so the per-pixel losses can be weighted by class before
    # they are reduced -- smp's cross-entropy takes no `weight` argument. Its
    # own "mean" is unusable here for a second reason anyway: it averages over
    # *all* pixels, counting the ignored ones as zero, which on mostly-unlabeled
    # patches would scale the supervised loss (and with it the effective
    # learning rate) toward zero.
    cross_entropy = smp.losses.SoftCrossEntropyLoss(
        reduction="none", smooth_factor=0.0, ignore_index=UNLABELED
    )
    # Neither overlap term takes CLASS_WEIGHTS, and neither needs to: both
    # average over the classes equally, so each class already contributes the
    # same regardless of how many pixels it has. Class weighting is a
    # cross-entropy concern only.
    dice = smp.losses.DiceLoss(
        mode="multiclass", ignore_index=UNLABELED, from_logits=True
    )
    # Lovasz-Softmax: a direct surrogate for IoU, averaged over the classes
    # actually present in the batch. Takes logits (it softmaxes internally).
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=UNLABELED)
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)

    def masked_ce(logits, target, mask, weighted=True):
        """Cross-entropy over the pixels `mask` selects (0.0 if none).

        Weighted by CLASS_WEIGHTS and normalized by the sum of the weights
        actually used, which is what makes the weights a reweighting rather than
        a rescaling: with equal weights this is exactly the plain mean over the
        masked pixels.

        `weighted=False` is for the pseudo-label term. Those targets are the
        model's own predictions, so up-weighting the target class there would
        feed the bias back into itself -- the model over-predicts target, is
        trained harder on its own target guesses, and over-predicts further.
        The supervised term is anchored to user labels, where the same weight is
        a controlled prior instead of a loop.
        """
        per_px = cross_entropy(logits, target).squeeze(1)  # (B, H, W), 0 if ignored
        if not weighted:
            return per_px.sum() / mask.sum().clamp(min=1)
        # target holds UNLABELED (255) outside the mask, which is not a valid
        # index; zero those out rather than indexing with them.
        safe = torch.where(mask, target, torch.zeros_like(target))
        per_px_weight = weights[safe] * mask
        return (per_px * per_px_weight).sum() / per_px_weight.sum().clamp(min=1e-6)

    # The convolutional encoder/decoder trains at LEARNING_RATE; a *transformer*
    # bottleneck gets the reduced BOTTLENECK_LR. Everything else is one group
    # because nothing here is pretrained, so there is no reason to stagger it.
    bottleneck = list(net.bottleneck.parameters())
    bottleneck_ids = {id(p) for p in bottleneck}
    rest = [p for p in net.parameters() if id(p) not in bottleneck_ids]
    param_groups = [
        {"params": rest, "lr": LEARNING_RATE},
        {"params": bottleneck, "lr": _bottleneck_lr(net.bottleneck)},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    rng = np.random.default_rng()
    gate_epoch = None
    stable_epochs = 0

    for epoch in range(epochs):
        if stop_event is not None and stop_event.is_set():
            return epoch
        net.train()
        warmup = gate_epoch is None
        if warmup:
            ramp = 0.0
        else:

            ramp_epochs = max(
                RAMP_MIN_EPOCHS, round(RAMP_FRACTION * (epochs - gate_epoch))
            )

            ramp = min(1.0, (epoch - gate_epoch + 1) / ramp_epochs)
        sums = {"sup": 0.0, "dice": 0.0, "lovasz": 0.0, "pseudo": 0.0, "cons": 0.0}
        n_batches = 0
        iou_inter = 0.0  # target-class intersection/union over labeled pixels,
        iou_union = 0.0  # accumulated across the epoch for a live IoU readout

        order = list(range(len(samples)))
        rng.shuffle(order)  # no fixed image ordering across epochs
        for idx in order:
            if stop_event is not None and stop_event.is_set():
                return epoch  # mid-epoch: this one does not count as done
            image, labels = samples[idx]()

            labels = labels.copy()
            labels[~image.valid_mask] = EXCLUDED
            for weak, strong, mask in _make_batches(
                image, labels, rng, skip_unlabeled=warmup
            ):
                strong, mask = strong.to(device), mask.to(device)

                logits = net(strong)  # (B, classes, H, W); mask is (B, H, W)

                sup_target = mask.masked_fill(mask == EXCLUDED, UNLABELED).long()
                labeled = sup_target != UNLABELED

                with torch.no_grad():
                    pred_t = (logits.argmax(1) == TARGET) & labeled
                    gt_t = (sup_target == TARGET) & labeled
                    iou_inter += (pred_t & gt_t).sum().item()
                    iou_union += (pred_t | gt_t).sum().item()
                # A batch can carry no user-labeled pixel at all: once the gate
                # opens, fully-unlabeled patches are the bulk of them. All three
                # supervised terms are then skipped together, because smp's
                # multiclass Lovasz returns an EMPTY tensor rather than a scalar
                # 0.0 when every pixel is ignored, and adding that to the total
                # makes the whole loss non-scalar -- backward() then raises.
                zero = logits.sum() * 0.0  # a real 0 that keeps the graph intact
                if labeled.any():
                    loss_sup = masked_ce(logits, sup_target, labeled)
                    loss_dice = dice(logits, sup_target) if LAMBDA_DICE else zero
                    loss_lovasz = lovasz(logits, sup_target) if LAMBDA_LOVASZ else zero
                else:
                    loss_sup = loss_dice = loss_lovasz = zero

                loss = (loss_sup + LAMBDA_DICE * loss_dice
                        + LAMBDA_LOVASZ * loss_lovasz)

                if not warmup:
                    weak = weak.to(device)
                    with torch.no_grad():
                        net.eval()
                        weak_probs = torch.softmax(net(weak), dim=1)
                        net.train()
                    strong_probs = torch.softmax(logits, dim=1)

                    # conf/pseudo are (B, H, W): the winning class and its
                    # probability, so the threshold keeps its meaning.
                    conf, pseudo = weak_probs.max(dim=1)
                    unlabeled = mask == UNLABELED
                    pseudo_mask = unlabeled & (conf > CONFIDENCE_THRESHOLD)
                    if pseudo_mask.any():
                        loss_pseudo = masked_ce(
                            logits,
                            pseudo.masked_fill(~pseudo_mask, UNLABELED),
                            pseudo_mask,
                            weighted=False,
                        )
                    else:
                        loss_pseudo = logits.sum() * 0.0

                    per_px = ((strong_probs - weak_probs) ** 2).mean(dim=1)
                    loss_cons = (
                        per_px[unlabeled].mean()
                        if unlabeled.any()
                        else logits.sum() * 0.0
                    )
                    loss = loss + ramp * (
                        LAMBDA_PSEUDO * loss_pseudo + LAMBDA_CONSISTENCY * loss_cons
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sums["sup"] += loss_sup.detach().item()
                sums["dice"] += loss_dice.detach().item()
                sums["lovasz"] += loss_lovasz.detach().item()
                if not warmup:
                    sums["pseudo"] += loss_pseudo.detach().item()
                    sums["cons"] += loss_cons.detach().item()
                n_batches += 1
            del image, labels  # bound memory: one image resident at a time

        denom = n_batches or 1
        # One supervised number, composed exactly like the batch loss, so the
        # figure the UI shows and the one the scheduler steps on are the same.
        loss_sup_epoch = (sums["sup"] + LAMBDA_DICE * sums["dice"]
                          + LAMBDA_LOVASZ * sums["lovasz"]) / denom


        if n_batches:
            scheduler.step(loss_sup_epoch)

        iou = iou_inter / iou_union if iou_union else None

        if gate_epoch is None:
            if iou is not None and iou >= TARGET_IOU_STABILITY:
                stable_epochs += 1
                if stable_epochs >= STABILITY_PATIENCE:
                    gate_epoch = epoch + 1  # the losses start on the next epoch
            else:
                stable_epochs = 0

        if progress is not None:
            progress(
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    # Reported as one supervised number, alongside pseudo/cons and
                    # the live target IoU.
                    "loss_sup": loss_sup_epoch,
                    "loss_pseudo": sums["pseudo"] / denom,
                    "loss_cons": sums["cons"] / denom,
                    "iou": round(iou, 4) if iou is not None else None,
                    "ramp": ramp,
                }
            )
    return epochs
