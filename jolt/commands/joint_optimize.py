"""Find 3D particle positions and jointly refine the tilt-series alignment (the main JOLT command).

For every tilt series in jolt.yaml, ``jolt joint_optimize`` builds 3D particle hypotheses from the 2D picks
(MLESAC), fills in missing tilt observations, runs bundle adjustment of the particle positions and per-tilt
shifts, and scores each particle with a figure of merit. It writes a 3D particle STAR file (plus an IMOD
.mod model for 3dmod), the refined IMOD .xf alignment and a run_status.json per tilt series.
"""

import argparse


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt joint_optimize`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("project_dir", help="JOLT project directory (created by jolt import)")

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--freeze_alignments",
        action="store_true",
        help="keep the per-tilt shifts fixed and optimize only the particle positions",
    )
    optimization.add_argument(
        "--shift_reg",
        type=float,
        default=0.001,
        help="penalty on changing the per-tilt shifts from the input alignment (lower = alignment moves more)",
    )
    optimization.add_argument(
        "--point_reg",
        type=float,
        default=0.0015,
        help="penalty on moving 3D particles from their initial positions (lower = particles move more)",
    )
    optimization.add_argument(
        "--residual_threshold",
        type=float,
        default=12.0,
        help="drop particles whose reprojection residual after bundle adjustment exceeds this (pixels)",
    )
    optimization.add_argument(
        "--mlesac_search_radius",
        type=float,
        default=12.0,
        help="radius for matching projected 3D hypotheses to 2D picks during initialization (pixels)",
    )
    optimization.add_argument("--seed", type=int, default=0, help="random seed")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--ts", default=None, help="comma-separated tilt-series names to process (default: all)")
    execution.add_argument("--gpus", default=None, help="GPU indices to use, e.g. 0,1 (default: all visible)")
    execution.add_argument(
        "--n_jobs", type=int, default=None, help="tilt series processed at the same time (default: one per GPU)"
    )
    execution.add_argument(
        "--worker_threads",
        type=int,
        default=1,
        help="CPU math threads (OpenMP/MKL/OpenBLAS) per tilt-series process",
    )
    execution.add_argument("--overwrite", action="store_true", help="redo tilt series that already have outputs")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt joint_optimize``."""
    raise NotImplementedError("`jolt joint_optimize` is not ported yet.")


if __name__ == "__main__":
    main(add_args().parse_args())
