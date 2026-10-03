"""Read a WarpTools project and create the JOLT project configuration (jolt.yaml).

``jolt import`` finds every tilt series (Warp ``.xml`` plus the ``tiltstack/<TS>/`` folder holding the
raw stack ``.st``, the IMOD ``.xf`` and the ``.tlt``), checks that the files agree with each other, and
writes ``<project_dir>/jolt.yaml``, which every later JOLT command reads.
"""

import argparse

from jolt.utils import HelpFormatter


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
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

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


def _check_picks(path, n_tilts: int) -> int:
    """Check that a 2D pick STAR file names tilt images by a stack index within range; return its pick count."""
    from jolt.io import read_star_table
    from jolt.pipeline import tilt_index_from_name

    table = read_star_table(path, required=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"])
    for name in table["rlnMicrographName"].astype(str).unique():
        idx = tilt_index_from_name(name)
        if not 1 <= idx <= n_tilts:
            raise ValueError(f"{path}: tilt image {name!r} has index {idx}, outside 1..{n_tilts}")
    return len(table)


def main(args: argparse.Namespace) -> None:
    """Run ``jolt import``."""
    from pathlib import Path

    from jolt.io import config, read_header_dims, warp
    from jolt.utils import parse_name_list

    project_dir = Path(args.project_dir)
    if config.config_path(project_dir).exists() and not args.overwrite:
        raise SystemExit(f"{config.config_path(project_dir)} already exists; pass --overwrite to replace it")
    warp_dir = Path(args.warp_dir).resolve()
    tomo_x, tomo_y, tomo_z = args.tomo_dims_xyz
    series = warp.discover(warp_dir, args.xml_path, args.tiltstack_path, parse_name_list(args.ts))

    problems, entries = [], {}
    for ts in series:
        entry = {
            "xml": str(ts.xml),
            "raw_stack": str(ts.raw_stack),
            "aligned_stack": str(ts.aligned_stack) if ts.aligned_stack else None,
            "xf": str(ts.xf),
            "tlt": str(ts.tlt),
            "n_tilts": ts.n_tilts,
            "raw_dims_xy": list(ts.raw_dims_xy),
            "pixel_size": ts.pixel_size,
        }
        if len(set(ts.tilt_angles)) != len(ts.tilt_angles):
            problems.append(f"{ts.name}: the .tlt lists the same tilt angle more than once")
        if ts.aligned_stack is not None:
            ax, ay, _ = read_header_dims(ts.aligned_stack)
            if (ax, ay) != (tomo_x, tomo_y):
                problems.append(
                    f"{ts.name}: --tomo_dims_xyz X,Y = {tomo_x},{tomo_y} but the aligned stack "
                    f"{ts.aligned_stack.name} is {ax} x {ay}; X,Y must be the aligned tilt-image size"
                )
        if args.picks_2d:
            pattern = Path(args.picks_2d.replace("{ts}", ts.name))
            picks = pattern if pattern.is_absolute() else warp_dir / pattern
            if not picks.is_file():
                problems.append(f"{ts.name}: 2D picks not found: {picks}")
            else:
                try:
                    _check_picks(picks, ts.n_tilts)
                    entry["picks_2d"] = str(picks)
                    entry["picks_frame"] = args.picks_frame
                except (ValueError, KeyError) as exc:
                    problems.append(str(exc))
        for warning in ts.warnings:
            print(f"WARNING: {warning}")
        entries[ts.name] = entry
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) found; nothing was written:\n  " + "\n  ".join(problems))

    xml_path, tiltstack_path = warp.default_paths(warp_dir)
    cfg = {
        "config_version": config.CONFIG_VERSION,
        "project": {
            "warp_dir": str(warp_dir),
            "xml_path": str(Path(args.xml_path).resolve()) if args.xml_path else str(xml_path),
            "tiltstack_path": str(Path(args.tiltstack_path).resolve()) if args.tiltstack_path else str(tiltstack_path),
            "tomo_dims_xyz": [int(tomo_x), int(tomo_y), int(tomo_z)],
        },
        "tilt_series": entries,
    }
    config.record_command(cfg, "import", {k: v for k, v in vars(args).items() if k != "func"})
    path = config.save(project_dir, cfg)
    n_picks = sum(1 for e in entries.values() if "picks_2d" in e)
    print(f"{len(entries)} tilt series imported ({n_picks} with 2D picks); configuration written to {path}")


if __name__ == "__main__":
    main(add_args().parse_args())
