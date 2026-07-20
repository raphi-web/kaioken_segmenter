"""Entry point: opens the pywebview window with the built React frontend.

Usage: python backend/main.py [path/to/image.tiff | path/to/project_dir]
"""

import os
import sys

import webview

if getattr(sys, "frozen", False):
    # PyInstaller bundle: backend/*.py are collected as top-level modules
    # (already importable) and frontend/dist is bundled under _MEIPASS.
    PROJECT_ROOT = sys._MEIPASS
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from api import Api

FRONTEND = os.path.join(PROJECT_ROOT, "frontend", "dist", "index.html")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "00.tiff")


def main():
    if not os.path.exists(FRONTEND):
        sys.exit("Frontend not built. Run: cd frontend && npm install && npm run build")
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and os.path.isdir(arg):
        api = Api(project_root=arg)
        title_suffix = os.path.basename(os.path.normpath(arg))
    else:
        image_path = arg or (DEFAULT_IMAGE if os.path.exists(DEFAULT_IMAGE) else None)
        api = Api(image_path=image_path)
        title_suffix = os.path.basename(image_path) if image_path else "no image"
    webview.create_window(
        f"Sentinel-2 Interactive Segmentation — {title_suffix}",
        FRONTEND,
        js_api=api,
        width=1500,
        height=950,
    )
    webview.start()


if __name__ == "__main__":
    main()
