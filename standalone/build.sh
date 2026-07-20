#!/usr/bin/env bash
# Build the standalone predictor executable with PyInstaller.
#
# Usage: ./build.sh
#
# Result: standalone/dist/predictor/  (predictor executable + _internal/)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_PYTHON="../venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "error: venv python not found at $VENV_PYTHON" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -m PyInstaller --version >/dev/null 2>&1; then
    echo "==> Installing PyInstaller into venv"
    "$VENV_PYTHON" -m pip install pyinstaller
fi

echo "==> Removing previous build/ and dist/"
rm -rf build dist

echo "==> Running PyInstaller"
"$VENV_PYTHON" -m PyInstaller predictor.spec --noconfirm

echo "==> Build complete: $(pwd)/dist/predictor/"
du -sh dist/predictor
