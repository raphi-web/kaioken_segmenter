"""Command line for the Kaio-ken Segmenter: `python -m kaioken <command>`."""

import argparse
import sys

from . import __version__


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="kaioken",
        description="Human-in-the-loop semantic segmentation for remote sensing.")
    parser.add_argument("--version", action="version", version=f"kaioken {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    start = sub.add_parser("start", help="open the labelling app")
    start.add_argument(
        "path", nargs="?",
        help="a GeoTIFF to label, or an existing project directory "
             "(default: an empty session)")

    fetch = sub.add_parser(
        "fetch-assets",
        help="download the pretrained weights and SAM2 models",
        description="Downloads into the per-user cache. Both are optional: without "
                    "the weights a model trains from scratch, without SAM2 the "
                    "click-assist button stays disabled.")
    fetch.add_argument("--weights", action="store_true",
                       help="pretrained encoder/decoder weights (23 MB)")
    fetch.add_argument("--sam2", action="store_true",
                       help="SAM2 encoder + decoder for click-assist (126 MB)")
    fetch.add_argument("--all", action="store_true",
                       help="everything (the default when no flag is given)")
    fetch.add_argument("--force", action="store_true",
                       help="re-download even if already cached")

    sub.add_parser(
        "predict", help="run a trained model over a GeoTIFF (no GUI)",
        add_help=False)  # its own parser owns --help; see _predict below

    sub.add_parser("paths", help="show where assets are being looked for")
    return parser


def _start(args):
    from .app import start
    start(args.path)
    return 0


def _fetch_assets(args):
    from . import assets

    names = []
    if args.weights or args.all:
        names.append("weights")
    if args.sam2 or args.all:
        names.extend(assets.SAM2_ASSETS)
    if not names:  # bare `fetch-assets` means everything
        names = list(assets.ASSETS)

    print(f"Downloading into {assets.cache_dir()}", file=sys.stderr)
    try:
        assets.fetch_all(names, force=args.force)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _predict(argv):
    """Hand the remaining arguments to the standalone predictor's own CLI."""
    from .predictor.cli import main as predictor_main
    return predictor_main(argv)


def _paths(_args):
    from . import assets

    root = assets.source_tree()
    print(f"source tree : {root or '(installed, not a checkout)'}")
    print(f"cache       : {assets.cache_dir()}")
    print(f"frontend    : {assets.frontend_index()}"
          f"{'' if assets.frontend_index().is_file() else '   [missing]'}")
    weights = assets.pretrained_weights()
    print(f"weights     : {weights}{'' if weights.is_file() else '   [missing]'}")
    sam2 = assets.sam2_onnx_dir()
    print(f"sam2 onnx   : {sam2}{'' if sam2.is_dir() else '   [missing]'}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    # `predict` forwards everything after it verbatim, so argparse never sees the
    # predictor's own flags and can't reject them.
    if argv and argv[0] == "predict":
        return _predict(argv[1:])

    args = parser.parse_args(argv)
    if args.command == "start":
        return _start(args)
    if args.command == "fetch-assets":
        return _fetch_assets(args)
    if args.command == "paths":
        return _paths(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
