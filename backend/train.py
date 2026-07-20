"""Semi-supervised training: supervised BCE + Lovasz + pseudo-labeling + consistency.

FixMatch-style scheme on patches (size from the image's data profile):
- a *weak* view (geometric augs, applied jointly to image and label mask) is
  forwarded without gradients to produce targets,
- a *strong* view (photometric noise on top of the weak view, so it stays
  pixel-aligned) is forwarded with gradients and receives all three losses.

User-labeled pixels are excluded from the pseudo-label and consistency masks,
so manual ground truth always overrides model beliefs.

The net emits one logit per pixel with sigmoid(logit) = P(target), so the
supervised term pairs per-pixel BCE with a Lovasz hinge, which optimizes the
target IoU directly; the pixel-wise losses keep BCE alone.
"""

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from data import UNLABELED

EXCLUDED = 254  # internal: source-nodata pixels take part in no loss at all
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.95
LAMBDA_LOVASZ = 1.0
LAMBDA_PSEUDO = 0.5
LAMBDA_CONSISTENCY = 1.0
WARMUP_EPOCHS = (
    10  # cumulative epochs of supervised-only training before pseudo/consistency
)
RAMP_EPOCHS = 5  # epochs *after* the warm-up to reach full unsupervised weight
MIN_EPOCHS = WARMUP_EPOCHS + RAMP_EPOCHS  # a run must cover both phases to be useful
SEMI_SUPERVISED_MIN_EPOCHS = 15  # shorter runs stay supervised-only (user labels only)

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
        A.ChannelDropout(channel_drop_range=(1, 2), fill=0.0, p=0.3),
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
TARGET = 0  # positive class in label space (0 target, 1 background)
P_MOSAIC = 0.2
P_CUTMIX = 0.2
P_COPY_PASTE = 0.2
CUTMIX_AREA = (0.05, 0.35)  # pasted box area, as a fraction of the patch
CUTMIX_ASPECT = (0.5, 2.0)
DONOR_ATTEMPTS = 8  # tries at finding a donor patch that meets a requirement


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
            )  # HWC for albumentations
            # Multi-patch augmentations first, so both views below come from the
            # same composed patch and stay pixel-aligned.
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


def train(model, samples, epochs, epochs_done=0, progress=None, stop_event=None):
    """Run the hybrid loop for `epochs` epochs over every labeled image.

    model: model.SegmentationModel;
    samples: sequence of zero-arg callables, each returning one
    (data.SentinelImage, labels (H, W) uint8) pair. They are called once per
    image per epoch and the result is dropped afterwards, so only a single
    image is resident at a time no matter how large the project is. Labels are
    in label space (= model class space): 0 target, 1 background, 255 unlabeled;
    epochs_done: cumulative epochs from previous runs (drives the warm-up and
    ramp-up); progress: callback(dict) after each epoch; stop_event:
    threading.Event. Returns the number of epochs actually completed, which is
    at least MIN_EPOCHS unless stopped early.

    An epoch is one shuffled pass over the patch grids of all samples, so the
    reported losses average over the whole labeled set.

    The first WARMUP_EPOCHS cumulative epochs are purely supervised: the
    unsupervised losses are gated off and fully-unlabeled patches are skipped,
    so the model first fits the explicit user labels. Afterwards pseudo-labeling
    and consistency fade in over RAMP_EPOCHS.

    A run whose selected length is below SEMI_SUPERVISED_MIN_EPOCHS is too short
    for pseudo-labeling/consistency to pay off, so it stays in that
    supervised-only mode for its whole duration, fitting the user-drawn labels
    alone.
    """
    if not samples:
        raise ValueError("No labeled images to train on")
    # Decided on the selected length, before the MIN_EPOCHS floor below (which
    # sits under the threshold, so the floor can't flip the decision).
    supervised_only = int(epochs) < SEMI_SUPERVISED_MIN_EPOCHS
    epochs = max(MIN_EPOCHS, int(epochs))
    net = model.net
    device = model.device
    # Lovasz collapses every ignored pixel into a single void id, so EXCLUDED is
    # folded into UNLABELED before it sees the mask.
    lovasz = smp.losses.LovaszLoss(mode="binary", ignore_index=UNLABELED)

    param_groups = [
        {"params": net.unet.encoder.parameters(), "lr": 1e-3},
        {"params": net.unet.decoder.parameters(), "lr": 2e-3},
        {"params": net.unet.segmentation_head.parameters(), "lr": 2e-3},
    ]
    if getattr(net, "point_head", None) is not None:
        # PointRend: the head learns through the refined logits scattered into
        # the map, so it just needs to be optimized alongside the U-Net.
        param_groups.append({"params": net.point_head.parameters(), "lr": 2e-3})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    rng = np.random.default_rng()

    for epoch in range(epochs):
        if stop_event is not None and stop_event.is_set():
            return epoch
        net.train()
        cumulative = epochs_done + epoch
        warmup = supervised_only or cumulative < WARMUP_EPOCHS
        # Rebased onto the end of the warm-up: on the old formula the ramp hit
        # 1.0 exactly as the gate opened, so the unsupervised losses arrived at
        # full weight in one step.
        ramp = (
            0.0
            if supervised_only
            else min(1.0, max(0.0, (cumulative + 1 - WARMUP_EPOCHS) / RAMP_EPOCHS))
        )
        sums = {"sup": 0.0, "lov": 0.0, "pseudo": 0.0, "cons": 0.0}
        n_batches = 0
        iou_inter = 0.0  # target-class intersection/union over labeled pixels,
        iou_union = 0.0  # accumulated across the epoch for a live IoU readout

        order = list(range(len(samples)))
        rng.shuffle(order)  # no fixed image ordering across epochs
        for idx in order:
            if stop_event is not None and stop_event.is_set():
                return epoch  # mid-epoch: this one does not count as done
            image, labels = samples[idx]()
            # Labels arrive in model class space; only source-nodata pixels are
            # excluded from every loss.
            labels = labels.copy()
            labels[~image.valid_mask] = EXCLUDED
            for weak, strong, mask in _make_batches(
                image, labels, rng, skip_unlabeled=warmup
            ):
                strong, mask = strong.to(device), mask.to(device)

                mask = mask.unsqueeze(
                    1
                )  # (B, 1, H, W), matching the single-logit output
                logits = net(strong)

                sup_target = mask.masked_fill(mask == EXCLUDED, UNLABELED)
                labeled = sup_target != UNLABELED
                # Target is the positive class, so flip out of label space (0 target).
                target = (sup_target == 0).float()
                # Live target IoU over the labeled pixels, read off the same
                # logits the supervised loss uses (logit > 0 <=> P(target) > 0.5),
                # so it costs nothing beyond a couple of masked reductions.
                with torch.no_grad():
                    pred_t = (logits > 0) & labeled
                    gt_t = (target > 0.5) & labeled
                    iou_inter += (pred_t & gt_t).sum().item()
                    iou_union += (pred_t | gt_t).sum().item()
                if labeled.any():
                    loss_sup = F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )[labeled].mean()
                else:
                    loss_sup = logits.sum() * 0.0
                loss_lov = lovasz(
                    logits.squeeze(1),
                    torch.where(labeled, target.long(), sup_target).squeeze(1),
                )

                loss = loss_sup + LAMBDA_LOVASZ * loss_lov

                # The weak view feeds only the unsupervised terms, so while they
                # are gated off it is never forwarded and the step costs one
                # forward pass instead of two.
                if not warmup:
                    weak = weak.to(device)
                    with torch.no_grad():
                        net.eval()
                        weak_probs = torch.sigmoid(net(weak))  # P(target)
                        net.train()
                    strong_probs = torch.sigmoid(logits)

                    conf = torch.maximum(weak_probs, 1.0 - weak_probs)
                    pseudo = (weak_probs > 0.5).float()
                    unlabeled = mask == UNLABELED
                    pseudo_mask = unlabeled & (conf > CONFIDENCE_THRESHOLD)
                    if pseudo_mask.any():
                        loss_pseudo = F.binary_cross_entropy_with_logits(
                            logits, pseudo, reduction="none"
                        )[pseudo_mask].mean()
                    else:
                        loss_pseudo = logits.sum() * 0.0

                    per_px = (strong_probs - weak_probs) ** 2
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
                sums["lov"] += loss_lov.detach().item()
                if not warmup:
                    sums["pseudo"] += loss_pseudo.detach().item()
                    sums["cons"] += loss_cons.detach().item()
                n_batches += 1
            del image, labels  # bound memory: one image resident at a time

        # A warm-up epoch over a project whose patches carry no annotation
        # at all skips every batch; report zeros rather than dividing by it.
        denom = n_batches or 1
        loss_sup_epoch = (sums["sup"] + LAMBDA_LOVASZ * sums["lov"]) / denom

        # Plateau is measured on the supervised loss alone, not the total: the
        # pseudo/consistency terms switch on at WARMUP_EPOCHS and ramp over the
        # next RAMP_EPOCHS, so the total rises there by composition rather than
        # by the model getting worse, and a plateau step would read that as
        # regression. An epoch that skipped every batch has no measurement at
        # all -- its 0.0 would look like a record improvement and reset the
        # patience counter -- so it does not step.
        if n_batches:
            scheduler.step(loss_sup_epoch)

        if progress is not None:
            # None when the epoch saw no labeled target pixels (e.g. only
            # background labels drawn), so the readout hides rather than shows 0.
            iou = iou_inter / iou_union if iou_union else None
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
