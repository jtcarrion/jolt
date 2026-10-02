"""Select 3D particles from a STAR file by figure of merit, residual, tilt series, and spacing.

``jolt filter`` applies only the gates you ask for (nothing is removed by default), writes the selected
particles to a new STAR file, and writes a summary of how many particles each gate removed.
"""

import argparse


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt filter`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", help="3D particle STAR file, or a jolt joint_optimize output directory")
    parser.add_argument("-o", "--output", required=True, help="output STAR file for the selected particles")

    gates = parser.add_argument_group("Selection (each gate is off unless given)")
    gates.add_argument("--min_fom", type=float, default=None, help="keep particles with figure of merit >= this")
    gates.add_argument(
        "--max_residual", type=float, default=None, help="keep particles with reprojection residual <= this (pixels)"
    )
    gates.add_argument(
        "--min_detector_score",
        type=float,
        default=None,
        help="keep particles whose mean 2D detector score is >= this (in the 2D picker's own units)",
    )
    gates.add_argument("--tilt_series", default=None, help="comma-separated tilt series to keep (default: all)")
    gates.add_argument("--exclude_tilt_series", default=None, help="comma-separated tilt series to drop")
    gates.add_argument(
        "--dedup_radius",
        type=float,
        default=0.0,
        help="remove near-duplicate particles closer than this, keeping the higher figure of merit (pixels; 0 = off)",
    )
    gates.add_argument(
        "--border", type=float, default=0.0, help="remove particles closer than this to any tomogram face (pixels)"
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt filter``."""
    raise NotImplementedError("`jolt filter` is not ported yet.")


if __name__ == "__main__":
    main(add_args().parse_args())
