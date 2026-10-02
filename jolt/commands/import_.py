"""Read a WarpTools project and create the JOLT project configuration (jolt.yaml).

``jolt import`` finds every tilt series (Warp ``.xml`` plus the ``tiltstack/<TS>/`` folder holding the
raw stack ``.st``, the IMOD ``.xf`` and the ``.tlt``), checks that the files agree with each other, and
writes ``<project_dir>/jolt.yaml``, which every later JOLT command reads.
"""

import argparse


def _xyz_dims(text: str) -> tuple[int, int, int]:
    """Parse ``"X,Y,Z"`` into three positive integers (pixels)."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected X,Y,Z (three integers), got {text!r}")
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected X,Y,Z (three integers), got {text!r}") from exc
    if min(dims) <= 0:
        raise argparse.ArgumentTypeError(f"tomogram dimensions must be positive, got {text!r}")
    return dims


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt import`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    core = parser.add_argument_group("Core arguments")
    core.add_argument("--warp_dir", required=True, help="WarpTools project directory")
    core.add_argument(
        "--project_dir",
        required=True,
        help="directory to create for this JOLT project; jolt.yaml and all later outputs are written here",
    )
    core.add_argument(
        "--tomo_dims_xyz",
        type=_xyz_dims,
        required=True,
        metavar="X,Y,Z",
        help="tomogram dimensions in pixels at the tilt-stack pixel size, the same dimensions used for "
        "alignment and reconstruction. X,Y must equal the aligned tilt-image size; Z (thickness) sets "
        "the 3D center",
    )

    paths = parser.add_argument_group("Path overrides (defaults are derived from --warp_dir)")
    paths.add_argument(
        "--xml_path",
        default=None,
        help="directory containing the Warp TS_*.xml files (default: <warp_dir>/warp_tiltseries)",
    )
    paths.add_argument(
        "--tiltstack_path",
        default=None,
        help="directory with one sub-directory per tilt series holding .st, .xf and .tlt "
        "(default: <warp_dir>/warp_tiltseries/tiltstack)",
    )

    picks = parser.add_argument_group("Existing 2D picks (skip jolt detect_2d)")
    picks.add_argument(
        "--picks_2d",
        default=None,
        help="path pattern of existing 2D pick STAR files, with {ts} replaced by the tilt-series name "
        "(e.g. picks/{ts}/{ts}_picks.star)",
    )
    picks.add_argument(
        "--picks_frame",
        choices=["raw", "aligned"],
        default="raw",
        help="frame of the --picks_2d coordinates: picked on the raw (unaligned) stack or on the aligned stack",
    )

    other = parser.add_argument_group("Other")
    other.add_argument("--ts", default=None, help="comma-separated tilt-series names to include (default: all)")
    other.add_argument("--overwrite", action="store_true", help="replace an existing jolt.yaml in --project_dir")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt import``."""
    raise NotImplementedError("`jolt import` is not ported yet.")


if __name__ == "__main__":
    main(add_args().parse_args())
