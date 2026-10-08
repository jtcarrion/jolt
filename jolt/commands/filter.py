"""Select 3D particles from a STAR file by figure of merit, residual, detector score, tilt series and spacing.

``jolt filter`` applies only the gates you ask for (nothing is removed by default), writes the selected
particles to a new STAR file, and writes a summary (``<output>.json``) of how many particles each gate removed.
The input is a particle STAR file or a ``jolt joint_optimize`` output directory (all ``*_particles.star`` files
in it are combined). Tilt series are identified by ``rlnMicrographName`` (``TS_001.tomostar`` -> ``TS_001``).
"""

import argparse

from jolt.utils import HelpFormatter


def _xyz_dims(text: str) -> tuple[int, int, int]:
    from jolt.commands.import_ import _xyz_dims as parse

    return parse(text)


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt filter`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

    parser.add_argument("input", help="3D particle STAR file, or a jolt joint_optimize output directory")
    parser.add_argument("-o", "--output", required=True, help="output STAR file for the selected particles")

    gates = parser.add_argument_group("Selection (each gate is off unless given)")
    gates.add_argument("--min_fom", type=float, default=None, help="keep particles with figure of merit >= this")
    gates.add_argument(
        "--max_residual",
        type=float,
        default=None,
        help="keep particles with reprojection residual (rlnJoltResidual: RMS over the matched tilts) <= this (pixels)",
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
        help="within each tilt series, remove near-duplicate particles closer than this, keeping the higher "
        "figure of merit (pixels; 0 = off)",
    )
    gates.add_argument(
        "--border", type=float, default=0.0, help="remove particles closer than this to any tomogram face (pixels)"
    )
    gates.add_argument(
        "--tomo_dims_xyz",
        type=_xyz_dims,
        default=None,
        metavar="X,Y,Z",
        help="tomogram dimensions in pixels, needed by --border (default: read from the jolt.yaml of the "
        "project the input belongs to)",
    )
    return parser


def _find_tomo_dims(start):
    """Return ``tomo_dims_xyz`` from the nearest jolt.yaml at or above ``start``, or None."""
    from pathlib import Path

    from jolt.io import config

    for folder in [Path(start).resolve(), *Path(start).resolve().parents]:
        if config.config_path(folder).is_file():
            return tuple(config.load(folder)["project"]["tomo_dims_xyz"])
    return None


def main(args: argparse.Namespace) -> None:
    """Run ``jolt filter``."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from jolt.io import read_star_table, write_star
    from jolt.selection import border_mask, dedup_indices
    from jolt.utils import parse_name_list

    source = Path(args.input)
    files = sorted(source.glob("*/*_particles.star")) if source.is_dir() else [source]
    if not files:
        raise SystemExit(f"no */*_particles.star files found under {source}")
    required = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    table = pd.concat([read_star_table(f, required=required) for f in files]).reset_index(drop=True)
    if "rlnMicrographName" in table.columns:
        ts_names = table["rlnMicrographName"].astype(str).str.replace(r"\.tomostar$", "", regex=True)
    else:
        ts_names = pd.Series([source.stem] * len(table))
    counts = {"input": int(len(table))}
    keep = np.ones(len(table), dtype=bool)

    def gate(name, mask):
        nonlocal keep
        before = int(keep.sum())
        keep &= np.asarray(mask, dtype=bool)
        counts[name] = before - int(keep.sum())

    def column(name, flag):
        if name not in table.columns:
            raise SystemExit(f"{flag} needs the column {name}, which the input does not have")
        return table[name].to_numpy(dtype=float)

    include = parse_name_list(args.tilt_series)
    exclude = parse_name_list(args.exclude_tilt_series)
    if include:
        gate("removed_tilt_series", ts_names.isin(include))
    if exclude:
        gate("removed_excluded_tilt_series", ~ts_names.isin(exclude))
    if args.min_fom is not None:
        gate("removed_min_fom", column("rlnAutopickFigureOfMerit", "--min_fom") >= args.min_fom)
    if args.max_residual is not None:
        gate("removed_max_residual", column("rlnJoltResidual", "--max_residual") <= args.max_residual)
    if args.min_detector_score is not None:
        score = column("rlnJoltDetectorScore", "--min_detector_score")
        gate("removed_min_detector_score", np.isfinite(score) & (score >= args.min_detector_score))
    xyz = table[required].to_numpy(dtype=float)
    if args.border > 0:
        dims = args.tomo_dims_xyz or _find_tomo_dims(source if source.is_dir() else source.parent)
        if dims is None:
            raise SystemExit("--border needs --tomo_dims_xyz (no jolt.yaml found above the input)")
        x, y, z = dims
        gate("removed_border", border_mask(xyz, (z, y, x), args.border))
    if args.dedup_radius > 0:
        fom_values = column("rlnAutopickFigureOfMerit", "--dedup_radius")
        dedup_keep = np.zeros(len(table), dtype=bool)
        for name in ts_names[keep].unique():
            rows = np.flatnonzero(keep & (ts_names == name).to_numpy())
            dedup_keep[dedup_indices(xyz, -fom_values, args.dedup_radius, rows)] = True
        gate("removed_dedup", dedup_keep)

    selected = table[keep].reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_star(output, selected)
    counts["kept"] = int(len(selected))
    summary = {
        "input": [str(f) for f in files],
        "parameters": {k: v for k, v in vars(args).items() if k != "func"},
        "counts": counts,
    }
    with open(output.with_suffix(".json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"kept {counts['kept']} of {counts['input']} particles -> {output}")
    for name, value in counts.items():
        if name.startswith("removed_"):
            print(f"  {name[8:]}: {value} removed")


if __name__ == "__main__":
    main(add_args().parse_args())
