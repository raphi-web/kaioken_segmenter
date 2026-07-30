"""Command-line entrypoint for the standalone predictor."""

import argparse
import os
import sys

from . import core


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="predictor",
        description="Predict a GeoTIFF with the exported U-Net (model.onnx).")
    parser.add_argument("input", help="input GeoTIFF")
    parser.add_argument("-o", "--output",
                        help="output GeoTIFF (default: <input>_prediction.tif)")
    parser.add_argument("-m", "--model",
                        help="model.onnx path (default: next to the executable)")
    parser.add_argument("-b", "--bands",
                        help="comma-separated 1-based band indices in model-channel "
                             "order (default: first N bands)")
    parser.add_argument("--nodata", type=float,
                        help="override the source nodata value")
    parser.add_argument("--probs", action="store_true",
                        help="also write a P(target) float raster")
    args = parser.parse_args(argv)

    model_path = args.model or core.default_model_path()
    if not os.path.exists(model_path):
        parser.error(f"model not found: {model_path} (pass --model)")
    if not os.path.exists(args.input):
        parser.error(f"input not found: {args.input}")

    band_map = None
    if args.bands:
        try:
            band_map = [int(b) for b in args.bands.split(",") if b.strip()]
        except ValueError:
            parser.error("--bands must be comma-separated integers, e.g. 1,2,3")

    output = args.output or (os.path.splitext(args.input)[0] + "_prediction.tif")

    def progress(done, total):
        print(f"\rtiles {done}/{total}", end="", flush=True)

    try:
        result = core.predict_geotiff(
            model_path, args.input, output, band_map=band_map,
            nodata=args.nodata, write_probs=args.probs, progress=progress)
    except (ValueError, OSError) as e:
        print()
        print(f"error: {e}", file=sys.stderr)
        return 1

    print()
    print(f"bands used (model-channel order): {result['band_map']}")
    print(f"wrote {result['outputs']['prediction']}")
    if "probability" in result["outputs"]:
        print(f"wrote {result['outputs']['probability']}")
    w, h = result["size"]
    print(f"target pixels: {result['target_pixels']:,} / {w * h:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
