"""PyInstaller entrypoint: dispatches to the predictor GUI or CLI.

Not imported by the app -- this is the script PyInstaller analyses when
"Export Executable" builds the standalone predictor. See _pyinstaller/predictor.spec.
"""

import sys

from kaioken.predictor.__main__ import main

if __name__ == "__main__":
    sys.exit(main() or 0)
