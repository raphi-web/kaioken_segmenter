"""Entrypoint dispatch: no arguments launches the GUI, arguments run the CLI.

This makes the packaged executable open a window when double-clicked, while
still being scriptable from a terminal.
"""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        from .cli import main as cli_main
        return cli_main(argv)
    from .gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
