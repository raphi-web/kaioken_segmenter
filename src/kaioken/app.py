"""Opens the pywebview window with the built React frontend.

Reached through `python -m kaioken start [path]`; see cli.py.
"""

import os
import sys

if sys.platform.startswith("linux"):
    _bundled_qt_themes = {"gtk3", "xdgdesktopportal"}
    _requested = os.environ.get("QT_QPA_PLATFORMTHEME", "")
    if not set(_requested.split(":")) & _bundled_qt_themes:
        os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal:gtk3"

import webview

from . import assets
from .api import Api


def default_image():
    """The repo's sample scene, when running from a checkout. Not shipped in the wheel."""
    root = assets.source_tree()
    if root is None:
        return None
    sample = root / "00.tiff"
    return str(sample) if sample.is_file() else None


def start(path=None):
    """Open the window on `path` (an image or a project dir), or on nothing."""
    frontend = assets.frontend_index()
    if not frontend.is_file():
        sys.exit(
            f"Frontend not found at {frontend}.\n"
            "Installed from a wheel? Reinstall -- the UI should be bundled.\n"
            "Running from a checkout? Build it: cd frontend && npm install && npm run build")

    if path and os.path.isdir(path):
        api = Api(project_root=path)
        title_suffix = os.path.basename(os.path.normpath(path))
    else:
        image_path = path or default_image()
        api = Api(image_path=image_path)
        title_suffix = os.path.basename(image_path) if image_path else "no image"

    webview.create_window(
        f"Sentinel-2 Interactive Segmentation — {title_suffix}",
        str(frontend),
        js_api=api,
        width=1500,
        height=950,
    )
    webview.start()
