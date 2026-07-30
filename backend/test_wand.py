"""Invariant tests for the magic wand (Api.wand_select).

    python3 backend/test_wand.py

Runs against a scratch COPY of the project in /tmp, never against the real
label_raster/. An earlier version of these tests wrote to the actual ground
truth and raced the running app's autosave; do not reintroduce that. The copy
takes three images -- one ordinary, one with nodata, one heavily fragmented --
which is everything the invariants need and 32 MB instead of 400.

Point the suite at a different project with WAND_TEST_PROJECT=/path/to/project.
When no project is found every test skips rather than fails: the invariants are
about real imagery, and a green run against synthetic noise would be a lie.

Label buffers are set explicitly with set_labels rather than read off disk. The
`protect` rules are about the value UNLABELED, and this dataset's masks are
int32 with -9999 for "no label", which _read_mask_file wraps to 241 -- a real
quirk of that data, but not one these tests should be measuring.
"""

import base64
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import Api  # noqa: E402
from data import UNLABELED  # noqa: E402
from sam_service import SamService  # noqa: E402

SOURCE_PROJECT = os.environ.get(
    "WAND_TEST_PROJECT", "/home/raphi/SSD-KHADAS/kaioken_labeler")
# One ordinary image, one with nodata, one fragmented. `nodata` is 14: it is the
# only image in this dataset with any, and the nodata test silently passed
# against 00 before it was pinned here.
IMAGES = {"plain": "00", "nodata": "14", "fragmented": "21"}

_scratch = None  # temp project root, built once for the module
_api = None


def setUpModule():
    global _scratch, _api
    if not os.path.isdir(os.path.join(SOURCE_PROJECT, "input_images")):
        return
    _scratch = tempfile.mkdtemp(prefix="wand_test_")
    os.makedirs(os.path.join(_scratch, "input_images"))
    os.makedirs(os.path.join(_scratch, "label_raster"))
    shutil.copy(os.path.join(SOURCE_PROJECT, "project_config.json"), _scratch)
    for stem in IMAGES.values():
        shutil.copy(os.path.join(SOURCE_PROJECT, "input_images", f"{stem}.tiff"),
                    os.path.join(_scratch, "input_images"))
        label = os.path.join(SOURCE_PROJECT, "label_raster", f"{stem}.tif")
        if os.path.exists(label):
            shutil.copy(label, os.path.join(_scratch, "label_raster"))
    _api = Api(project_root=_scratch)


def tearDownModule():
    if _scratch and os.path.isdir(_scratch):
        shutil.rmtree(_scratch, ignore_errors=True)


class WandTestCase(unittest.TestCase):
    """Shared fixture: the scratch project, a loaded image, mask helpers."""

    IMAGE = "plain"

    @classmethod
    def setUpClass(cls):
        if _api is None:
            raise unittest.SkipTest(
                f"no test project at {SOURCE_PROJECT} (set WAND_TEST_PROJECT)")

    def setUp(self):
        self.api = _api
        result = self.api.load_image(f"{IMAGES[self.IMAGE]}.tiff")
        self.assertTrue(result.get("ok", True), result)
        self.image = self.api._image
        self.h, self.w = self.image.height, self.image.width
        # A blank buffer unless a test sets its own, so `protect` starts inert
        # and one test's labels cannot leak into the next.
        self.set_labels(np.full((self.h, self.w), UNLABELED, np.uint8))

    def set_labels(self, labels):
        self.api.set_labels(base64.b64encode(labels.tobytes()).decode("ascii"))

    def select(self, samples, **options):
        options.setdefault("tolerance", 0.13)
        result = self.api.wand_select(samples, options)
        self.assertTrue(result["ok"], result.get("error"))
        return result

    def mask(self, result):
        bits = np.unpackbits(np.frombuffer(base64.b64decode(result["mask"]), np.uint8))
        return bits[:self.h * self.w].reshape(self.h, self.w).astype(bool)

    def select_mask(self, samples, **options):
        return self.mask(self.select(samples, **options))

    def components(self, mask):
        """Number of 4-connected components in a boolean mask."""
        import cv2
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return count - 1  # label 0 is the background

    # Two clicks on materially different ground, so the union is a real union.
    A = [256, 256]
    B = [400, 120]


# ---------------------------------------------------------------- contract


class TestOptionsContract(WandTestCase):

    def test_error_without_an_image(self):
        """No image loaded is a message, not an exception across the bridge."""
        blank = Api.__new__(Api)
        blank._image = None
        self.assertEqual(Api.wand_select(blank, [self.A], {"tolerance": 0.1}),
                         {"ok": False, "error": "No image loaded"})

    def test_error_without_samples(self):
        for samples in ([], None):
            with self.subTest(samples=samples):
                result = self.api.wand_select(samples, {"tolerance": 0.1})
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"], "No samples given")

    def test_every_bad_option_is_a_message(self):
        bad = [
            ({}, "A numeric tolerance is required"),
            ({"tolerance": "wide"}, "A numeric tolerance is required"),
            ({"tolerance": None}, "A numeric tolerance is required"),
            ({"tolerance": 0.1, "sample_size": "big"}, "sample_size must be an integer"),
            ({"tolerance": 0.1, "max_pixels": "many"},
             "max_pixels must be an integer or null"),
            ({"tolerance": 0.1, "max_pixels": 0}, "max_pixels must be positive"),
            ({"tolerance": 0.1, "max_pixels": -5}, "max_pixels must be positive"),
            ({"tolerance": 0.1, "source": "vibes"}, "Unknown wand source: vibes"),
            ({"tolerance": 0.1, "level": "deepest"}, "Unknown feature level: deepest"),
        ]
        for options, message in bad:
            with self.subTest(options=options):
                result = self.api.wand_select([self.A], options)
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"], message)
        # Malformed coordinates too, since they arrive straight from JS.
        self.assertFalse(self.api.wand_select([["a", "b"]], {"tolerance": 0.1})["ok"])

    def test_mask_payload_matches_the_image(self):
        """count, width/height and the packed mask all describe the same array."""
        result = self.select([self.A, self.B])
        self.assertEqual((result["height"], result["width"]), (self.h, self.w))
        mask = self.mask(result)
        self.assertEqual(int(mask.sum()), result["count"])
        self.assertEqual(mask.shape, (self.h, self.w))
        self.assertEqual(result["valid"], int(self.image.valid_mask.sum()))

    def test_sample_size_is_clamped_not_rejected(self):
        """Out-of-range sizes clamp into 1..25 so a stale UI cannot error."""
        for size in (-4, 0, 1, 25, 900):
            with self.subTest(sample_size=size):
                self.assertTrue(self.select([self.A], sample_size=size)["ok"])


# ---------------------------------------------- the additivity guarantee


class TestAdditivity(WandTestCase):
    """ADDING A POSITIVE CLICK NEVER REMOVES A PIXEL.

    Tested in its strongest form: the selection for a click list is EXACTLY the
    union of the selections of its individual clicks. That is stricter than
    additivity and it is what makes additivity hold -- each sample is resolved
    independently, so nothing about one can reach another.
    """

    def assert_independent(self, samples, **options):
        combined = self.select_mask(samples, **options)
        union = np.zeros_like(combined)
        for sample in samples:
            union |= self.select_mask([sample], **options)
        np.testing.assert_array_equal(combined, union)
        return combined

    def test_independent_across_tolerances(self):
        for tolerance in (0.05, 0.09, 0.13, 0.19, 0.30):
            with self.subTest(tolerance=tolerance):
                self.assert_independent([self.A, self.B], tolerance=tolerance)

    def test_independent_across_budgets(self):
        for budget in (None, 100, 2_000, 50_000):
            with self.subTest(max_pixels=budget):
                self.assert_independent([self.A, self.B], max_pixels=budget)

    def test_adding_a_click_never_removes_a_pixel(self):
        """The property as a user experiences it: keep clicking, lose nothing."""
        clicks = [self.A]
        grown = self.select_mask(clicks)
        for extra in ([440, 300], [60, 460], [200, 90]):
            before, clicks = grown, clicks + [extra]
            grown = self.select_mask(clicks)
            self.assertEqual(int((before & ~grown).sum()), 0,
                             f"adding {extra} removed pixels")
            self.assertGreaterEqual(int(grown.sum()), int(before.sum()))

    def test_a_sample_is_always_inside_its_own_selection(self):
        """A click is never a silent no-op, even below its own threshold."""
        for tolerance in (0.0, 0.001, 0.13):
            for x, y in ([self.A, self.B]):
                with self.subTest(tolerance=tolerance, point=(x, y)):
                    mask = self.select_mask([[x, y]], tolerance=tolerance)
                    self.assertTrue(mask[y, x])

    def test_independent_with_every_option_at_once(self):
        labels = np.full((self.h, self.w), UNLABELED, np.uint8)
        labels[:, :150] = 3
        self.set_labels(labels)
        self.assert_independent(
            [self.A, self.B], tolerance=0.16, max_pixels=4_000,
            negatives=[[480, 480]], protect=True)


# ------------------------------------------------------------ the budget


class TestBudget(WandTestCase):

    def test_budget_is_exact_and_flagged(self):
        budget = 750
        result = self.select([self.A], max_pixels=budget, tolerance=0.3)
        self.assertTrue(result["capped"])
        self.assertEqual(result["count"], budget)

    def test_available_reports_the_uncapped_size(self):
        uncapped = self.select([self.A], tolerance=0.3)
        capped = self.select([self.A], tolerance=0.3, max_pixels=500)
        self.assertEqual(capped["available"], uncapped["count"])
        self.assertLess(capped["count"], capped["available"])
        self.assertFalse(uncapped["capped"])

    def test_budget_is_per_sample_not_pooled(self):
        """A second click must not eat the first one's allowance."""
        budget = 600
        alone = self.select([self.A], max_pixels=budget, tolerance=0.3)
        together = self.select([self.A, self.B], max_pixels=budget, tolerance=0.3)
        self.assertEqual(alone["count"], budget)
        # Two samples get two budgets; a shared pot would leave ~one.
        self.assertGreater(together["count"], budget)
        self.assertLessEqual(together["count"], 2 * budget)
        # And the first sample keeps every pixel it had on its own.
        self.assertEqual(int((self.mask(alone) & ~self.mask(together)).sum()), 0)

    def test_a_clipped_region_is_still_connected(self):
        """BFS growth, not nearest-N: the trimmed region must not fall apart."""
        mask = self.select_mask([self.A], max_pixels=800, tolerance=0.3)
        self.assertEqual(self.components(mask), 1)

    def test_budget_above_the_region_changes_nothing(self):
        plain = self.select([self.A])
        roomy = self.select([self.A], max_pixels=plain["count"] + 10_000)
        self.assertFalse(roomy["capped"])
        self.assertEqual(roomy["count"], plain["count"])


# ------------------------------------------------------- connectivity


class TestConnectivity(WandTestCase):

    def test_normal_mode_returns_one_piece(self):
        for tolerance in (0.09, 0.13, 0.19):
            with self.subTest(tolerance=tolerance):
                self.assertEqual(self.components(self.select_mask([self.A],
                                                                  tolerance=tolerance)), 1)

    def test_four_connected_not_eight(self):
        """Two blocks touching at one corner are two regions, not one.

        Driven through a synthetic field so the geometry is exact: with
        8-connectivity the fill would cross the shared corner and take both.
        """
        field = np.ones((self.h, self.w), np.float32)
        field[10:20, 10:20] = 0.0   # the clicked block
        field[20:30, 20:30] = 0.0   # touches it only at pixel (20, 20)'s corner
        original = self.api._wand._spectral_field
        self.api._wand._spectral_field = lambda y, x, n: field
        try:
            self.api._clear_wand_cache()
            mask = self.select_mask([[15, 15]], tolerance=0.5)
        finally:
            self.api._wand._spectral_field = original
            self.api._clear_wand_cache()
        self.assertTrue(mask[10:20, 10:20].all())
        self.assertFalse(mask[20:30, 20:30].any())

    def test_nodata_is_never_selected(self):
        """Only 14.tiff has any in this dataset, hence the pinned image."""
        self.api.load_image(f"{IMAGES['nodata']}.tiff")
        image = self.api._image
        invalid = ~image.valid_mask
        self.assertGreater(int(invalid.sum()), 0, "14.tiff should carry nodata")
        self.h, self.w = image.height, image.width
        mask = self.select_mask([self.A], tolerance=0.4, **{"global": True})
        self.assertEqual(int((mask & invalid).sum()), 0)


# ------------------------------------------------------------ global mode


class TestGlobalMode(WandTestCase):
    IMAGE = "fragmented"

    def test_global_reaches_past_the_clicked_component(self):
        local = self.select([self.A], tolerance=0.13)
        everywhere = self.select([self.A], tolerance=0.13, **{"global": True})
        self.assertGreater(everywhere["count"], local["count"])
        self.assertGreater(self.components(self.mask(everywhere)), 1)
        # Global is a superset: it is the same match set without the bound.
        self.assertEqual(int((self.mask(local) & ~self.mask(everywhere)).sum()), 0)

    def test_global_budget_takes_the_nearest_pixels(self):
        budget = 500
        capped = self.select_mask([self.A], tolerance=0.2, max_pixels=budget,
                                  **{"global": True})
        full = self.select_mask([self.A], tolerance=0.2, **{"global": True})
        self.assertEqual(int(capped.sum()), budget)
        # Every kept pixel is at least as close to the click as every dropped one.
        ys, xs = np.nonzero(capped)
        dy, dx = np.nonzero(full & ~capped)
        kept = ((ys - self.A[1]) ** 2 + (xs - self.A[0]) ** 2).max()
        dropped = ((dy - self.A[1]) ** 2 + (dx - self.A[0]) ** 2).min()
        self.assertLessEqual(kept, dropped)

    def test_global_is_still_additive(self):
        combined = self.select_mask([self.A, self.B], tolerance=0.12,
                                    **{"global": True})
        for sample in (self.A, self.B):
            alone = self.select_mask([sample], tolerance=0.12, **{"global": True})
            self.assertEqual(int((alone & ~combined).sum()), 0)


# -------------------------------------------------------------- protection


class TestProtection(WandTestCase):

    def stripe_labels(self, value=3):
        """A labeled band straight across the middle of the image."""
        labels = np.full((self.h, self.w), UNLABELED, np.uint8)
        labels[self.h // 2 - 8:self.h // 2 + 8, :] = value
        self.set_labels(labels)
        return labels

    def test_protection_is_a_strict_subset(self):
        self.stripe_labels()
        plain = self.select_mask([self.A], tolerance=0.2, **{"global": True})
        guarded = self.select_mask([self.A], tolerance=0.2, protect=True,
                                   **{"global": True})
        self.assertEqual(int((guarded & ~plain).sum()), 0)
        self.assertLess(int(guarded.sum()), int(plain.sum()))

    def test_protection_reports_what_it_withheld(self):
        self.stripe_labels()
        plain = self.select([self.A], tolerance=0.2, **{"global": True})
        guarded = self.select([self.A], tolerance=0.2, protect=True,
                              **{"global": True})
        self.assertEqual(guarded["protected"], plain["count"] - guarded["count"])
        self.assertEqual(plain["protected"], 0)

    def test_protection_conducts_through_labeled_pixels(self):
        """A labeled strip must not sever the fill; it just is not painted.

        Uses a synthetic field: an unbroken bar crossing the labeled stripe, so
        anything on the far side is reachable only THROUGH labeled ground.
        """
        mid = self.h // 2
        field = np.ones((self.h, self.w), np.float32)
        field[mid - 60:mid + 60, 100:130] = 0.0  # a bar spanning the stripe
        self.stripe_labels()
        original = self.api._wand._spectral_field
        self.api._wand._spectral_field = lambda y, x, n: field
        try:
            self.api._clear_wand_cache()
            mask = self.select_mask([[115, mid - 50]], tolerance=0.5, protect=True)
        finally:
            self.api._wand._spectral_field = original
            self.api._clear_wand_cache()
        far_side = mask[mid + 20:mid + 60, 100:130]
        self.assertTrue(far_side.any(), "the fill did not cross the labeled stripe")
        self.assertFalse(mask[mid - 8:mid + 8, 100:130].any(),
                         "labeled pixels were painted anyway")

    def test_protection_and_global_together(self):
        self.stripe_labels()
        guarded = self.select_mask([self.A, self.B], tolerance=0.2, protect=True,
                                   **{"global": True})
        labeled = self.api._labels != UNLABELED
        self.assertEqual(int((guarded & labeled).sum()), 0)
        self.assertGreater(int(guarded.sum()), 0)


# --------------------------------------------------------------- negatives


class TestNegatives(WandTestCase):
    IMAGE = "fragmented"

    def test_a_negative_shrinks_the_selection(self):
        plain = self.select_mask([self.A], tolerance=0.25, **{"global": True})
        ys, xs = np.nonzero(plain)
        far = int(np.argmax((ys - self.A[1]) ** 2 + (xs - self.A[0]) ** 2))
        cut = self.select_mask([self.A], tolerance=0.25, **{"global": True},
                               negatives=[[int(xs[far]), int(ys[far])]])
        self.assertEqual(int((cut & ~plain).sum()), 0)
        self.assertLess(int(cut.sum()), int(plain.sum()))
        self.assertFalse(cut[ys[far], xs[far]])

    def test_a_negative_reaches_every_fragment(self):
        """The rule is nearest-neighbour, not local: one click cuts everywhere.

        Synthetic field: the clicked blob plus two disconnected blobs of the
        same material. Right-clicking one of them must remove the other too.
        """
        field = np.ones((self.h, self.w), np.float32)
        for y0, x0 in ((40, 40), (200, 200), (300, 60)):
            field[y0:y0 + 20, x0:x0 + 20] = 0.0
        other = np.ones((self.h, self.w), np.float32)
        other[200:220, 200:220] = 0.0
        other[300:320, 60:80] = 0.0
        fields = {(50, 50): field, (210, 210): other, (310, 70): other}
        original = self.api._wand._spectral_field
        self.api._wand._spectral_field = lambda y, x, n: fields[(y, x)]
        try:
            self.api._clear_wand_cache()
            plain = self.select_mask([[50, 50]], tolerance=0.5, **{"global": True})
            cut = self.select_mask([[50, 50]], tolerance=0.5, **{"global": True},
                                   negatives=[[210, 210]])
        finally:
            self.api._wand._spectral_field = original
            self.api._clear_wand_cache()
        self.assertTrue(plain[300:320, 60:80].any())
        # The negative was clicked on one fragment; the OTHER one goes too.
        self.assertFalse(cut[200:220, 200:220].any())
        self.assertFalse(cut[300:320, 60:80].any())
        self.assertTrue(cut[40:60, 40:60].any(), "the positive click survives")

    def test_negatives_with_global_protect_and_budget(self):
        labels = np.full((self.h, self.w), UNLABELED, np.uint8)
        labels[:120, :] = 5
        self.set_labels(labels)
        result = self.select([self.A, self.B], tolerance=0.2, max_pixels=3_000,
                             protect=True, negatives=[[20, 20]], **{"global": True})
        mask = self.mask(result)
        self.assertEqual(int((mask & (labels != UNLABELED)).sum()), 0)
        self.assertLessEqual(result["count"], 2 * 3_000)
        self.assertEqual(int(mask.sum()), result["count"])


# ------------------------------------------------------- embedding source


@unittest.skipUnless(SamService().available(), "sam2/onnx models not present")
class TestEmbeddingSource(WandTestCase):
    """The SAM2-feature source. Skipped wholesale when the models are absent."""

    SAM = {"source": "sam"}

    def test_an_unknown_level_is_a_message(self):
        for options in ({"source": "sam", "level": "shallow"},
                        {"source": "embedding"}):
            with self.subTest(options=options):
                result = self.api.wand_select([self.A], {"tolerance": 0.8, **options})
                self.assertFalse(result["ok"])
                self.assertIn("Unknown", result["error"])

    def test_the_field_is_bounded_finite_and_image_sized(self):
        """Shape is the one thing that could silently drift from the label buffer:
        the field is computed on a smaller grid and upsampled."""
        for level in ("fine", "mid", "deep"):
            with self.subTest(level=level):
                field = self.api._wand._embedding_field(self.A[1], self.A[0], 9, level)
                self.assertEqual(field.shape, (self.h, self.w))
                self.assertTrue(np.isfinite(field).all())
                self.assertGreaterEqual(float(field.min()), 0.0)
                self.assertLessEqual(float(field.max()), 2.0)

    def test_the_two_sources_are_genuinely_different(self):
        """Guards against the embedding path quietly falling back to spectral."""
        spectral = self.api._wand._spectral_field(self.A[1], self.A[0], 9)
        embedded = self.api._wand._embedding_field(self.A[1], self.A[0], 9, "fine")
        self.assertFalse(np.allclose(spectral, embedded))
        self.assertNotEqual(
            int(self.select_mask([self.A], tolerance=0.13).sum()),
            int(self.select_mask([self.A], tolerance=0.8, **self.SAM).sum()))

    def test_adding_a_click_never_removes_a_pixel(self):
        combined = self.select_mask([self.A, self.B], tolerance=0.85, **self.SAM)
        union = np.zeros_like(combined)
        for sample in (self.A, self.B):
            union |= self.select_mask([sample], tolerance=0.85, **self.SAM)
        np.testing.assert_array_equal(combined, union)

    def test_deterministic_budgeted_and_connected(self):
        first = self.select_mask([self.A], tolerance=0.85, **self.SAM)
        again = self.select_mask([self.A], tolerance=0.85, **self.SAM)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(self.components(first), 1)
        capped = self.select([self.A], tolerance=0.85, max_pixels=300, **self.SAM)
        self.assertTrue(capped["capped"])
        self.assertEqual(capped["count"], 300)

    def test_the_shipped_default_tolerance_is_usable(self):
        """0.8 (slider 50) must land where the source does something.

        Fails if a re-exported encoder moves the distances, which is the point:
        the useful band is narrow and the default has to sit inside it.
        """
        result = self.select([self.A], tolerance=0.8, **self.SAM)
        share = result["count"] / result["valid"]
        self.assertGreater(share, 0.0005, f"default selects almost nothing ({share:.4%})")
        self.assertLess(share, 0.25, f"default floods the image ({share:.2%})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
