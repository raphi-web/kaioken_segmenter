"""Project configuration (project_config.json) and the .thumbnails cache.

A project is a folder of GeoTIFF images plus a config file describing the
data profile, the class palette and where user/AI masks are stored. Paths in
the config are relative to the project root.
"""

import hashlib
import json
import os
import re
import threading

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

CONFIG_NAME = "project_config.json"
THUMB_DIR = ".thumbnails"
THUMB_SIZE = 128
IMAGE_EXTENSIONS = (".tif", ".tiff")

# Label space everywhere (UI, masks, model classes): 0 target, 1 background,
# 255 unlabeled (data.UNLABELED, not a class).
DEFAULT_CLASSES = [
    {"id": 0, "name": "Target", "color": "#ff5028"},
    {"id": 1, "name": "Background", "color": "#3c8cff"},
]

# Band names of the original 10-band Sentinel-2 stacks, used as the default
# naming when a project has exactly 10 bands.
SENTINEL2_BAND_NAMES = ["Red", "Green", "Blue", "NIR1", "RE1", "RE2", "RE3",
                        "NIR2", "SWIR1", "SWIR2"]

# Share of the images held out of training to score the model on (see
# effective_validation). Fixed when the project is created; individual images
# are moved between the sets afterwards via the split overrides.
DEFAULT_VALIDATION_RATIO = 0.2
MAX_VALIDATION_RATIO = 0.9  # a project must keep something to train on
SPLIT_ROLES = ("training", "validation")


def default_band_names(input_channels):
    if input_channels == len(SENTINEL2_BAND_NAMES):
        return list(SENTINEL2_BAND_NAMES)
    return [f"Band {i + 1}" for i in range(input_channels)]


def default_display_bands(input_channels):
    """0-based [R, G, B] band indices used for the on-screen RGB composite:
    the first three bands, with the last repeated when fewer than three exist."""
    idx = list(range(min(3, input_channels)))
    while len(idx) < 3:
        idx.append(idx[-1] if idx else 0)
    return idx


def split_rank(name):
    """A stable pseudo-random score in [0, 1) derived from an image's name.

    Keyed on the name rather than the image's position in the sorted list, so
    adding or removing images leaves every other image's rank untouched: a new
    image flips at most one existing assignment (the one at the cutoff).
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2.0**64


def deterministic_validation(names, ratio):
    """The int(ratio * N) lowest-ranked names, as a set (the base split)."""
    n_val = int(len(names) * ratio)
    if n_val <= 0:
        return set()
    # Name breaks rank ties so the result never depends on input ordering.
    ordered = sorted(names, key=lambda n: (split_rank(n), n))
    return set(ordered[:n_val])


def default_split():
    return {"validation_ratio": DEFAULT_VALIDATION_RATIO, "overrides": {}}


def effective_validation(names, split):
    """Sorted validation image names: the base split with overrides applied.

    Membership is computed, never stored, so images appearing (GeoTIFF tiling)
    or disappearing on disk need no reconciliation pass. Only the user's manual
    per-image deviations live in the config, as `overrides`.
    """
    split = split or default_split()
    ratio = split.get("validation_ratio", DEFAULT_VALIDATION_RATIO)
    overrides = split.get("overrides") or {}
    base = deterministic_validation(names, ratio)
    validation = {n for n in names if overrides.get(n, "") != "training"
                  and (n in base or overrides.get(n) == "validation")}
    return sorted(validation)


def default_config(project_name, input_channels=10, patch_size=96):
    return {
        "project_name": project_name,
        "data_profile": {"input_channels": input_channels,
                         "input_patch_size": [patch_size, patch_size],
                         "band_names": default_band_names(input_channels),
                         "display_bands": default_display_bands(input_channels),
                         "use_pointrend": False},
        "classes": [dict(c) for c in DEFAULT_CLASSES],
        "split": default_split(),
        "paths": {"images_folder": ".", "masks_user": "masks_user", "masks_ai": "masks_ai"},
    }


def config_from_settings(settings):
    """Config dict from the frontend's project-setup form (raises ValueError)."""
    try:
        channels = int(settings["input_channels"])
        size = int(settings["input_patch_size"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid project config: bands and patch size must be integers")
    names = settings.get("band_names") or default_band_names(channels)
    raw_display = settings.get("display_bands")
    if raw_display:
        try:
            display = [min(max(int(b), 0), channels - 1) for b in raw_display][:3]
        except (TypeError, ValueError):
            raise ValueError("Invalid project config: display_bands must be integers")
        while len(display) < 3:
            display.append(display[-1] if display else 0)
    else:
        display = default_display_bands(channels)
    raw_ratio = settings.get("validation_ratio", DEFAULT_VALIDATION_RATIO)
    try:
        ratio = round(float(DEFAULT_VALIDATION_RATIO if raw_ratio is None else raw_ratio), 3)
    except (TypeError, ValueError):
        raise ValueError("Invalid project config: validation_ratio must be a number")
    ratio = min(max(ratio, 0.0), MAX_VALIDATION_RATIO)
    config = {
        "project_name": str(settings.get("project_name", "")).strip() or "project",
        "data_profile": {
            "input_channels": channels,
            "input_patch_size": [size, size],
            "band_names": [str(n).strip() for n in names],
            "display_bands": display,
            "use_pointrend": bool(settings.get("use_pointrend", False)),
        },
        "classes": [dict(c) for c in DEFAULT_CLASSES],
        # Overrides are per-image and only exist once the user moves one, so a
        # freshly built config always starts from the plain ratio.
        "split": {"validation_ratio": ratio, "overrides": {}},
        "paths": {
            "images_folder": str(settings.get("images_folder", "")).strip() or ".",
            "masks_user": str(settings.get("masks_user", "")).strip() or "masks_user",
            "masks_ai": str(settings.get("masks_ai", "")).strip() or "masks_ai",
        },
    }
    return validate_config(config)


def probe_band_count(folder):
    """Band count of the first readable GeoTIFF in folder, or None."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS))
    except FileNotFoundError:
        return None
    for name in names:
        try:
            with rasterio.open(os.path.join(folder, name)) as src:
                return src.count
        except Exception:
            continue
    return None


def validate_config(config):
    """Return the config if it matches the schema, else raise ValueError."""
    def expect(condition, message):
        if not condition:
            raise ValueError(f"Invalid project config: {message}")

    expect(isinstance(config, dict), "root must be a JSON object")
    expect(isinstance(config.get("project_name"), str) and config["project_name"].strip(),
           "'project_name' must be a non-empty string")

    profile = config.get("data_profile")
    expect(isinstance(profile, dict), "'data_profile' must be an object")
    expect(isinstance(profile.get("input_channels"), int) and profile["input_channels"] > 0,
           "'data_profile.input_channels' must be a positive integer")
    patch = profile.get("input_patch_size")
    expect(isinstance(patch, list) and len(patch) == 2
           and all(isinstance(v, int) and v > 0 for v in patch),
           "'data_profile.input_patch_size' must be [int, int]")
    expect(patch[0] == patch[1],
           "'data_profile.input_patch_size' must be square, e.g. [96, 96]")
    expect(patch[0] % 32 == 0,
           "'data_profile.input_patch_size' must be a multiple of 32 "
           "(the encoder downsamples 32x)")
    band_names = profile.get("band_names")
    if band_names is not None:  # optional: older configs don't have it
        expect(isinstance(band_names, list)
               and all(isinstance(n, str) and n.strip() for n in band_names),
               "'data_profile.band_names' must be a list of non-empty strings")
        expect(len(band_names) == profile["input_channels"],
               "'data_profile.band_names' must have one name per input channel")

    use_pointrend = profile.get("use_pointrend")
    if use_pointrend is not None:  # optional: older configs default to off
        expect(isinstance(use_pointrend, bool),
               "'data_profile.use_pointrend' must be a boolean")

    display_bands = profile.get("display_bands")
    if display_bands is not None:  # optional: older configs default to first three
        expect(isinstance(display_bands, list) and len(display_bands) == 3
               and all(isinstance(b, int) and 0 <= b < profile["input_channels"]
                       for b in display_bands),
               "'data_profile.display_bands' must be three band indices in "
               "[0, input_channels)")

    classes = config.get("classes")
    expect(isinstance(classes, list) and classes, "'classes' must be a non-empty list")
    seen_ids = set()
    for i, cls in enumerate(classes):
        expect(isinstance(cls, dict), f"classes[{i}] must be an object")
        expect(isinstance(cls.get("id"), int) and 0 <= cls["id"] <= 254,
               f"classes[{i}].id must be an integer in [0, 254]")
        expect(cls["id"] not in seen_ids, f"duplicate class id {cls['id']}")
        seen_ids.add(cls["id"])
        expect(isinstance(cls.get("name"), str) and cls["name"].strip(),
               f"classes[{i}].name must be a non-empty string")
        expect(isinstance(cls.get("color"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}", cls["color"]),
               f"classes[{i}].color must be a '#rrggbb' string")

    split = config.get("split")
    if split is not None:  # optional: older configs default to default_split()
        expect(isinstance(split, dict), "'split' must be an object")
        ratio = split.get("validation_ratio")
        expect(isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
               and 0.0 <= ratio <= MAX_VALIDATION_RATIO,
               f"'split.validation_ratio' must be a number in "
               f"[0, {MAX_VALIDATION_RATIO}]")
        overrides = split.get("overrides")
        if overrides is not None:
            expect(isinstance(overrides, dict), "'split.overrides' must be an object")
            for name, role in overrides.items():
                expect(isinstance(name, str) and name.strip(),
                       "'split.overrides' keys must be non-empty image names")
                expect(role in SPLIT_ROLES,
                       f"split.overrides['{name}'] must be one of {SPLIT_ROLES}")

    paths = config.get("paths")
    expect(isinstance(paths, dict), "'paths' must be an object")
    for key in ("images_folder", "masks_user", "masks_ai"):
        value = paths.get(key)
        expect(isinstance(value, str) and value.strip(), f"'paths.{key}' must be a non-empty string")
        expect(not os.path.isabs(value), f"'paths.{key}' must be relative to the project root")
    return config


def config_path(root):
    return os.path.join(root, CONFIG_NAME)


def load_config(root):
    with open(config_path(root), encoding="utf-8") as f:
        return validate_config(json.load(f))


def save_config(root, config):
    """Atomically write the validated config; returns its path."""
    validate_config(config)
    path = config_path(root)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


class Project:
    """An open project: config, resolved folders and thumbnail cache state."""

    def __init__(self, root, config):
        self.root = os.path.abspath(root)
        self.config = config
        self.thumb_progress = {"running": False, "done": 0, "total": 0}
        self._thumb_thread = None
        for folder in (self.masks_user_dir, self.masks_ai_dir, self.thumb_dir):
            os.makedirs(folder, exist_ok=True)

    @classmethod
    def create(cls, root):
        """New project rooted at an image directory, with default config.

        input_channels is inferred from the first image found so the model
        matches the project's data (e.g. 3-band RGB or 4-band RGBN rasters).
        """
        name = os.path.basename(os.path.normpath(root)) or "project"
        channels = probe_band_count(root) or 10
        return cls(root, default_config(name, input_channels=channels))

    @classmethod
    def open(cls, root):
        return cls(root, load_config(root))

    def _resolve(self, relative):
        return os.path.normpath(os.path.join(self.root, relative))

    @property
    def images_dir(self):
        return self._resolve(self.config["paths"]["images_folder"])

    @property
    def masks_user_dir(self):
        return self._resolve(self.config["paths"]["masks_user"])

    @property
    def masks_ai_dir(self):
        return self._resolve(self.config["paths"]["masks_ai"])

    @property
    def thumb_dir(self):
        return os.path.join(self.root, THUMB_DIR)

    @property
    def model_checkpoint_path(self):
        """The project's trained-model checkpoint (weights + epoch counter)."""
        return os.path.join(self.root, "model.pt")

    @property
    def input_channels(self):
        return self.config["data_profile"]["input_channels"]

    @property
    def patch_size(self):
        return self.config["data_profile"]["input_patch_size"][0]

    @property
    def display_bands(self):
        """0-based [R, G, B] band indices for previews, or None (first three)."""
        return self.config["data_profile"].get("display_bands")

    @property
    def use_pointrend(self):
        """Whether the model refines uncertain pixels with the PointRend head."""
        return self.config["data_profile"].get("use_pointrend", False)

    @property
    def split(self):
        return self.config.get("split") or default_split()

    @property
    def validation_ratio(self):
        return self.split.get("validation_ratio", DEFAULT_VALIDATION_RATIO)

    def validation_names(self):
        """Sorted names of the images held out of training (see effective_validation)."""
        return effective_validation(self.list_images(), self.split)

    def is_validation(self, name):
        return name in set(self.validation_names())

    def image_role(self, name):
        return "validation" if self.is_validation(name) else "training"

    def list_images(self):
        try:
            return sorted(name for name in os.listdir(self.images_dir)
                          if name.lower().endswith(IMAGE_EXTENSIONS))
        except FileNotFoundError:
            return []

    def image_path(self, name):
        return os.path.join(self.images_dir, name)

    def mask_paths(self, image_name):
        """(user_mask_path, ai_mask_path) for an image, always .tif."""
        base = os.path.splitext(image_name)[0] + ".tif"
        return os.path.join(self.masks_user_dir, base), os.path.join(self.masks_ai_dir, base)

    def thumb_path(self, image_name):
        return os.path.join(self.thumb_dir, os.path.splitext(image_name)[0] + ".png")

    def start_thumbnail_generation(self, force=False):
        """Generate thumbnails on a background thread.

        Existing files are kept unless force=True, which regenerates (overwrites)
        every thumbnail with the project's current display bands — used when the
        RGB display mapping changes.
        """
        if self._thumb_thread and self._thumb_thread.is_alive():
            return
        names = self.list_images()
        self.thumb_progress = {"running": True, "done": 0, "total": len(names)}
        self._thumb_thread = threading.Thread(target=self._thumb_worker,
                                              args=(names, force), daemon=True)
        self._thumb_thread.start()

    def _thumb_worker(self, names, force=False):
        bands = self.display_bands
        for i, name in enumerate(names):
            path = self.thumb_path(name)
            if force or not os.path.exists(path):
                try:
                    generate_thumbnail(self.image_path(name), path, bands=bands)
                except Exception:
                    pass  # unreadable image: skip, leave no thumbnail
            self.thumb_progress["done"] = i + 1
        self.thumb_progress["running"] = False


def generate_thumbnail(image_path, thumb_path, size=THUMB_SIZE, bands=None):
    """Write an RGB PNG preview (bounded by size x size).

    `bands` is a 3-list of 0-based band indices mapped to R, G, B (the project's
    display_bands); when omitted it uses the first three bands (last repeated if
    fewer). Decimated reading keeps this cheap for large rasters; each channel is
    percentile-stretched like the main viewer so thumbnails match the panes.
    """
    with rasterio.open(image_path) as src:
        scale = size / max(src.width, src.height)
        out_h = max(1, round(src.height * scale))
        out_w = max(1, round(src.width * scale))
        if bands:
            idx = [min(max(int(b), 0), src.count - 1) for b in bands][:3]
        else:
            idx = list(range(min(3, src.count)))
        while len(idx) < 3:
            idx.append(idx[-1] if idx else 0)
        data = src.read([i + 1 for i in idx], out_shape=(3, out_h, out_w),
                        resampling=Resampling.average).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    valid = np.all(data != nodata, axis=0)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for b in range(3):
        band = data[b][valid]
        if band.size:
            lo, hi = np.percentile(band, (2, 98))
            if hi <= lo:
                hi = lo + 1.0
            rgb[..., b] = (np.clip((data[b] - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    rgb[~valid] = 0

    tmp = thumb_path + ".tmp"
    Image.fromarray(rgb).save(tmp, format="PNG")
    os.replace(tmp, thumb_path)  # atomic: readers never see a half-written file
