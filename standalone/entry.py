"""PyInstaller entrypoint: dispatches to the predictor GUI or CLI."""

import sys

from predictor.__main__ import main

if __name__ == "__main__":
    sys.exit(main() or 0)
