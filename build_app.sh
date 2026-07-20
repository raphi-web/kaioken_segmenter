#!/usr/bin/env bash
# Build the full Kaio-ken Segmenter desktop app with PyInstaller.
#
# Usage: ./build_app.sh
#
# Result: dist/kaioken-segmenter/  (executable + _internal/)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_PYTHON="venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    echo "==> Installing PyInstaller into venv"
    "$VENV_PYTHON" -m pip install pyinstaller
fi

if [[ ! -f frontend/dist/index.html ]]; then
    echo "==> Frontend not built, running npm install && npm run build"
    (cd frontend && npm install && npm run build)
fi

if [[ ! -x standalone/dist/predictor/predictor ]]; then
    echo "==> Standalone predictor not built, running standalone/build.sh"
    (cd standalone && ./build.sh)
fi

echo "==> Removing previous build/ and dist/"
rm -rf build dist

echo "==> Running PyInstaller"
"$VENV_PYTHON" -m PyInstaller app.spec --noconfirm

echo "==> Bundling standalone predictor for in-app 'Export Executable'"
mkdir -p dist/kaioken-segmenter/standalone
cp -r standalone/dist/predictor dist/kaioken-segmenter/standalone/predictor

if [[ -f sam2/onnx/sam2.1_hiera_tiny.encoder.onnx || -f sam2/onnx/sam2.1_hiera_small.encoder.onnx ]]; then
    echo "==> Bundling sam2/onnx for in-app SAM2 assist"
    mkdir -p dist/kaioken-segmenter/sam2
    cp -r sam2/onnx dist/kaioken-segmenter/sam2/onnx
else
    echo "==> sam2/onnx not found; SAM2 assist will stay disabled in the packaged app"
    echo "    (regenerate with backend/export_sam_onnx.py, see its docstring)"
fi

echo "==> Build complete: $(pwd)/dist/kaioken-segmenter/"
du -sh dist/kaioken-segmenter
