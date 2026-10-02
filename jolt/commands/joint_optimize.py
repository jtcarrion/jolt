"""Find 3D particle positions and jointly refine the tilt-series alignment (the main JOLT command).

For every tilt series in jolt.yaml that has 2D picks, ``jolt joint_optimize`` builds 3D particle hypotheses from
the picks (MLESAC), runs bundle adjustment of the particle positions and per-tilt shifts, selects the particles
and scores each one with a figure of merit. Per tilt series it writes, in ``<project_dir>/joint_optimize/<TS>/``
(or under ``--output_dir``): the 3D particle STAR file (pixels at the tilt-stack pixel size, with figure of merit,
mean 2D detector score and reprojection residual), an IMOD .mod model for 3dmod, the refined IMOD .xf alignment
and run_status.json.
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
    optimization.add_argument(
        "--match_distance",
        type=float,
        default=12.0,
        help="bundle adjustment matches each projected particle to its nearest pick within this distance (pixels)",
    )
    optimization.add_argument(
        "--fom_search_radius",
        type=float,
        default=10.0,
        help="radius of the soft pick assignment used by the figure of merit and the detector score (pixels)",
    )

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--ts", default=None, help="comma-separated tilt-series names to process (default: all)")
    execution.add_argument("--gpus", default=None, help="GPU indices to use, e.g. 0,1, or 'cpu' (default: all visible)")
    execution.add_argument(
        "--n_jobs", type=int, default=None, help="tilt series processed at the same time (default: one per GPU)"
    )
    execution.add_argument(
        "--worker_threads",
        type=int,
        default=1,
        help="CPU math threads (OpenMP/MKL/OpenBLAS) per tilt-series process",
    )
    execution.add_argument(
        "--output_dir", default=None, help="where to write the results (default: <project_dir>/joint_optimize)"
    )
    execution.add_argument("--overwrite", action="store_true", help="redo tilt series that already have outputs")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt joint_optimize``."""
    import json
    from pathlib import Path

    from jolt import batch, pipeline
    from jolt.bundle_adjustment import BundleAdjustmentSettings
    from jolt.io import config
    from jolt.mlesac import MlesacSettings
    from jolt.utils import FileLog, parse_name_list

    project_dir = Path(args.project_dir)
    cfg = config.load(project_dir)
    series = cfg["tilt_series"]
    names = parse_name_list(args.ts) or sorted(series)
    unknown = [n for n in names if n not in series]
    if unknown:
        raise SystemExit(f"tilt series not in {config.config_path(project_dir)}: {unknown}")
    no_picks = [n for n in names if not series[n].get("picks_2d")]
    if no_picks:
        raise SystemExit(f"no 2D picks registered for {no_picks}; run jolt detect_2d or jolt import --picks_2d")
    out_root = Path(args.output_dir) if args.output_dir else project_dir / "joint_optimize"
    done = [n for n in names if (out_root / n / "run_status.json").exists()]
    if done and not args.overwrite:
        raise SystemExit(f"{len(done)} tilt series already have outputs in {out_root} ({done[:5]}...); use --overwrite")

    settings = pipeline.JointOptimizeSettings(
        mlesac=MlesacSettings(search_radius_px=args.mlesac_search_radius),
        bundle=BundleAdjustmentSettings(
            shift_reg=args.shift_reg,
            point_reg=args.point_reg,
            max_match_distance=args.match_distance,
            freeze_shifts=args.freeze_alignments,
        ),
        residual_threshold=args.residual_threshold,
        fom_search_radius=args.fom_search_radius,
    )
    x, y, z = cfg["project"]["tomo_dims_xyz"]
    tasks = []
    for name in names:
        entry = series[name]
        inputs = pipeline.TiltSeriesInputs(
            name=name,
            picks_2d=Path(entry["picks_2d"]),
            picks_frame=entry.get("picks_frame", "raw"),
            xf=Path(entry["xf"]),
            tlt=Path(entry["tlt"]),
            raw_dims_xy=tuple(entry["raw_dims_xy"]),
            tomo_dims_xyz=(x, y, z),
        )
        tasks.append(batch.Task(name, pipeline.run_tilt_series_logged, (inputs, str(out_root / name), settings)))

    out_root.mkdir(parents=True, exist_ok=True)
    log = FileLog(out_root / "joint_optimize.log")
    gpus = batch.visible_gpus(args.gpus)
    where = "GPUs " + ", ".join(gpus) if gpus else "CPU"
    log(f"joint_optimize: {len(tasks)} tilt series on {where} ({args.n_jobs or 'auto'} jobs)")
    outcomes = batch.run_tasks(tasks, gpus, args.n_jobs, args.worker_threads, log=log)

    summary = {}
    for outcome in sorted(outcomes, key=lambda o: o["name"]):
        if outcome["ok"]:
            summary[outcome["name"]] = outcome["result"]
        else:
            summary[outcome["name"]] = {"error": outcome["error"], "traceback": outcome.get("traceback")}
    with open(out_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    not_ok = {
        n: s for n, s in summary.items() if "error" in s or s.get("ba_status") != "ok" or s.get("fom_status") != "ok"
    }
    for name, s in not_ok.items():
        log(
            f"ATTENTION {name}: "
            + (s["error"] if "error" in s else f"bundle adjustment {s['ba_status']}, FOM {s['fom_status']}")
        )
    n_particles = sum(s.get("n_particles", 0) for s in summary.values())
    log(
        f"{n_particles} particles from {len(summary) - sum('error' in s for s in summary.values())} tilt series; "
        f"{len(not_ok)} need attention; summary in {out_root / 'summary.json'}"
    )

    parameters = {k: v for k, v in vars(args).items() if k != "func"}
    config.record_command(cfg, "joint_optimize", parameters, {"directory": str(out_root.resolve())})
    config.save(project_dir, cfg)
    if any("error" in s for s in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main(add_args().parse_args())
