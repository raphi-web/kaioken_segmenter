"""Magic-wand selection: region growing from clicks, over image bands or SAM2 features.

The whole design serves one property:

    ADDING A POSITIVE CLICK NEVER REMOVES A PIXEL.

Each sample is resolved completely independently -- its own reference, its own
match set, its own connected region, its own pixel budget -- and the results are
unioned, so nothing about one sample can reach another. Negative clicks are the
deliberate exception; removing pixels is what they are for.

Split from api.py, which keeps the parts that are genuinely bridge concerns: the
options contract and its error strings, the SAM2 availability check, bit-packing
the mask for transport. This module is the algorithm, and it neither imports nor
knows about pywebview.

`api.Api` holds one WandSelector for its lifetime, binds it to the active image
per call and clears it when the pixels change; see WandSelector.bind / .clear.
"""

import numpy as np
from data import UNLABELED
from sam_service import resize_bilinear

try:
    import cv2  # only for the connected-component pass; see _connected_region
except ImportError:  # pragma: no cover - opencv is a project dependency
    cv2 = None

# Offsets of a 5x5 window ordered nearest-first, so _sample_colour can take the
# N pixels closest to a click by slicing the front of this list. The click
# itself is offset 0, hence always included.
_SAMPLE_OFFSETS = sorted(
    ((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)),
    key=lambda o: (o[0] * o[0] + o[1] * o[1], o[0], o[1]),
)
MAX_SAMPLE_PIXELS = len(_SAMPLE_OFFSETS)  # 25

# What "similar material" can be measured against. Both share every line of the
# machinery below except the distance itself.
SOURCES = ("image", "sam")


class WandResult:
    """One selection: the mask plus the counts the UI needs to explain it.

    `available` is what the selection would have been with no budget, so the UI
    can say how hard the cap is biting; `protected` is what `protect` withheld.
    """

    __slots__ = ("mask", "available", "protected", "capped")

    def __init__(self, mask, available, protected, capped):
        self.mask = mask
        self.available = available
        self.protected = protected
        self.capped = capped


class WandSelector:
    """Resolves wand clicks against one bound image, caching per-sample fields.

    Holds two caches, both valid only for the currently bound image:

      _field_cache   (source, level, y, x, sample_size) -> full-image distance field
      _stack_cache   level -> centred unit-length SAM2 features, so each new
                     sample is one dot product rather than a re-prepare

    Adding a click therefore costs one new field, dragging the tolerance slider
    costs none, and flipping between the two sources costs nothing after the
    first look at each.
    """

    def __init__(self, sam):
        self._sam = sam
        self._image = None
        self._image_name = None
        self._field_cache = {}
        self._stack_cache = {}

    # ---------- binding / cache lifetime ----------

    def bind(self, image, image_name):
        """Point the selector at an image, dropping caches if it is a new one.

        The image is held rather than passed to every helper so that the field
        methods keep the (y, x, sample_size) signatures the algorithm is written
        in. Api serializes wand calls under its own lock, so there is never more
        than one binding in flight.
        """
        if image_name != self._image_name:
            self.clear()
        self._image = image
        self._image_name = image_name

    def clear(self):
        """Drop every cached field. Called whenever the pixels change.

        Both caches describe one specific image, and a display-band change keeps
        the name while replacing the pixels the SAM2 encoder reads -- so this
        cannot be left to bind()'s name check alone.
        """
        self._field_cache.clear()
        self._stack_cache.clear()

    # ---------- reference and distance ----------

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
        if level not in self._stack_cache:
            feats = np.asarray(self._sam.features(level), dtype=np.float32)
            feats = feats - feats.mean(axis=(1, 2), keepdims=True)
            norm = np.linalg.norm(feats, axis=0, keepdims=True)
            self._stack_cache[level] = feats / np.maximum(norm, 1e-8)
        return self._stack_cache[level]

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

    def _fields(self, samples, sample_size, source, level):
        """[((y, x), field), ...] -- one full-image distance field per sample.

        `source` picks how distance is measured and nothing else in the wand
        changes with it: connectivity, the budget, the negatives and protection
        all consume only the field.
        """
        out = []
        for y, x in samples:
            key = (source, level, y, x, sample_size)
            if key not in self._field_cache:
                self._field_cache[key] = (
                    self._embedding_field(y, x, sample_size, level)
                    if source == "sam"
                    else self._spectral_field(y, x, sample_size))
            out.append(((y, x), self._field_cache[key]))
        return out

    # ---------- region and budget ----------

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
            return WandSelector._grow_within_budget(within, seed, within.size)
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

    def _region(self, field, seed, tolerance, negatives, is_global, budget):
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

    # ---------- the selection ----------

    def select(self, points, negatives, tolerance, sample_size, source, level,
               is_global, budget, labels=None, protect=False):
        """Resolve every sample independently and union the results.

        Coordinates are (y, x) in image pixels and are assumed already validated
        and clamped -- that is the bridge's job (see api.Api.wand_select), which
        is also where a bad option becomes a message rather than an exception.
        """
        image = self._image
        fields = self._fields(points, sample_size, source, level)
        # Each negative is compared at ITS OWN distance to a pixel, so the
        # threshold a pixel must beat is the negative's field value there -- not
        # a scalar. These are therefore full fields.
        neg_fields = [f for _, f in
                      self._fields(negatives, sample_size, source, level)]

        selected = np.zeros((image.height, image.width), dtype=bool)
        available = np.zeros_like(selected)
        capped = False
        for seed, field in fields:
            region, uncapped, hit = self._region(
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
        if protect and labels is not None:
            labeled = selected & (labels != UNLABELED)
            protected = int(labeled.sum())
            selected &= ~labeled
        return WandResult(selected, int(available.sum()), protected, capped)
