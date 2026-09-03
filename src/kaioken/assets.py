"""Where the app's files live, whether it runs from a wheel or a git checkout.

Small files (the built frontend) ride along inside the wheel. The model assets do
not: the SAM2 encoder alone is 109 MB, so shipping them would make every install
pay for features many users never touch. They are downloaded on demand instead,
into a per-user cache -- `python -m kaioken fetch-assets`.

Every resolver returns a path that may not exist. That is deliberate: each caller
already treats a missing asset as "feature off" rather than an error (see
model.load_pretrained, SamService.available).

Lookup order is the same everywhere:

1. An environment variable, so a user can point at their own copy.
2. The download cache.
3. The source tree, when running from a checkout -- keeps `python -m kaioken start`
   in a clone working off the repo's own pretraining/ and sam2/ as it always has.
"""

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from platformdirs import user_cache_dir

# Release holding the downloadable assets. They are versioned separately from the
# package: the weights change far less often than the code, and re-uploading
# 130 MB for a patch release would be silly.
ASSET_RELEASE = "https://github.com/raphi-web/kaioken_segmenter/releases/download/assets-v1"


class Asset:
    """One downloadable file: where it goes, where it comes from, what it should be."""

    def __init__(self, name, relpath, sha256, size, description):
        self.name = name
        self.relpath = relpath  # below the cache root
        self.sha256 = sha256
        self.size = size
        self.description = description

    @property
    def url(self):
        return f"{ASSET_RELEASE}/{Path(self.relpath).name}"

    @property
    def path(self):
        return cache_dir() / self.relpath


ASSETS = {
    "weights": Asset(
        "weights",
        "weights/pretrained.pth",
        "6a283c2c04129ccb7f17be67b05aa4481c6a30ddc2fe408cbd28a644e901a6ec",
        23200593,
        "Pretrained encoder/decoder weights (default initialization)",
    ),
    "sam2-encoder": Asset(
        "sam2-encoder",
        "sam2/onnx/sam2.1_hiera_tiny.encoder.onnx",
        "bd89ae8775db73f6cfd5cd698be62178d14ca4b0afe30026a64d30ef07eb5871",
        109384618,
        "SAM2 image encoder (click-assist)",
    ),
    "sam2-decoder": Asset(
        "sam2-decoder",
        "sam2/onnx/sam2.1_hiera_tiny.decoder.onnx",
        "7905ff10f54e7b667f24661e16e90ede716b32bdaf019b400ae09140f7c4ad86",
        16513172,
        "SAM2 mask decoder (click-assist)",
    ),
}

SAM2_ASSETS = ("sam2-encoder", "sam2-decoder")


def cache_dir():
    """Per-user download cache (~/.cache/kaioken on Linux)."""
    return Path(os.environ.get("KAIOKEN_CACHE_DIR") or user_cache_dir("kaioken"))


def source_tree():
    """The repo root when running from a checkout, else None.

    Identified by a pyproject.toml next to the frontend/ sources *whose*
    `src/kaioken` is this very package. Matching on the marker files alone is not
    enough: a venv created inside the checkout puts site-packages below the repo
    root, so an installed wheel would walk up into the repo and quietly serve its
    assets -- exactly the confusion this fallback exists to avoid.
    """
    package_dir = Path(__file__).resolve().parent
    for parent in package_dir.parents:
        if not ((parent / "pyproject.toml").is_file() and (parent / "frontend").is_dir()):
            continue
        return parent if (parent / "src" / "kaioken") == package_dir else None
    return None


def frontend_index():
    """index.html of the built React UI.

    Unlike the model assets this one is required, and it is bundled in the wheel,
    so a miss means either a broken build or an unbuilt checkout.

    In a checkout frontend/dist wins over the bundled copy. The build hook leaves
    a _frontend/ behind in the source tree, and preferring it would show a
    developer their last packaged UI instead of the `npm run build` they just ran.
    """
    root = source_tree()
    if root is not None:
        return root / "frontend" / "dist" / "index.html"
    return Path(__file__).resolve().parent / "_frontend" / "index.html"


def pretrained_weights():
    """Default weights for a fresh model, or a nonexistent path if unavailable."""
    override = os.environ.get("KAIOKEN_WEIGHTS")
    if override:
        return Path(override)
    cached = ASSETS["weights"].path
    if cached.is_file():
        return cached
    root = source_tree()
    if root is not None:
        return root / "pretraining" / "pretrained.pth"
    return cached


def sam2_onnx_dir():
    """Directory holding the SAM2 encoder/decoder graphs."""
    override = os.environ.get("KAIOKEN_SAM2_DIR")
    if override:
        return Path(override)
    cached = cache_dir() / "sam2" / "onnx"
    if cached.is_dir():
        return cached
    root = source_tree()
    if root is not None:
        return root / "sam2" / "onnx"
    return cached


# ---------- download ----------


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_present(asset):
    """Whether the asset is already downloaded and intact."""
    return asset.path.is_file() and asset.path.stat().st_size == asset.size


def fetch(asset, on_progress=None, force=False):
    """Download one asset into the cache, verify it, return its path.

    Streams to a .part file and renames only after the digest checks out, so an
    interrupted download can never masquerade as a usable model.
    """
    dest = asset.path
    if is_present(asset) and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(asset.url) as response:
            total = int(response.headers.get("Content-Length") or asset.size)
            done = 0
            with open(part, "wb") as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {asset.name} ({asset.url}): {e}") from e

    actual = _digest(part)
    if actual != asset.sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {asset.name}: expected {asset.sha256}, got {actual}")
    shutil.move(part, dest)
    return dest


def fetch_all(names, on_progress=None, force=False, stream=sys.stderr):
    """Download several assets, reporting progress as a one-line-per-asset log."""
    paths = []
    for name in names:
        asset = ASSETS[name]
        if is_present(asset) and not force:
            print(f"  {asset.name}: already present ({asset.path})", file=stream)
            paths.append(asset.path)
            continue
        print(f"  {asset.name}: {asset.description}", file=stream)

        def report(done, total, _a=asset):
            pct = 100 * done / total if total else 0
            print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)",
                  end="", file=stream, flush=True)

        paths.append(fetch(asset, on_progress=on_progress or report, force=force))
        print(f"\r    done -> {asset.path}{' ' * 20}", file=stream)
    return paths
