"""Pick 2D particles on every raw tilt image with Topaz and add the picks to jolt.yaml.

``jolt detect_2d`` splits each raw tilt stack into tilt images, runs Topaz (preprocessing and extraction,
seeded for reproducibility), writes one 2D pick STAR file per tilt series, and records the pick files in
jolt.yaml for ``jolt joint_optimize``. Skip it if you registered your own picks with
``jolt import --picks_2d``.
"""

import argparse

from jolt.utils import HelpFormatter


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt detect_2d`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

    parser.add_argument("project_dir", help="JOLT project directory (created by jolt import)")

    picking = parser.add_argument_group("Picking")
    picking.add_argument(
        "--radius",
        type=int,
        required=True,
        help="Topaz extraction radius in pixels of the tilt stack (about the particle radius); "
        "two picks are never closer than this",
    )
    selection = picking.add_mutually_exclusive_group()
    selection.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="keep picks with a Topaz score above this (default: -6 when --top_k is not given)",
    )
    selection.add_argument("--top_k", type=int, default=None, help="keep the best K picks on each tilt image")
    picking.add_argument(
        "--weights",
        default=None,
        help="Topaz model: a pretrained name (e.g. resnet16) or a .sav file "
        "(default: the jolt finetune weights recorded in jolt.yaml if present, else resnet16)",
    )
    picking.add_argument("--seed", type=int, default=0, help="seed for Topaz preprocessing (same seed, same picks)")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--ts", default=None, help="comma-separated tilt-series names to process (default: all)")
    execution.add_argument("--devices", default=None, help="GPU indices to use, e.g. 0,1, or 'cpu' (default: all)")
    execution.add_argument(
        "--topaz_bin", default=None, help="Topaz executable (default: $JOLT_TOPAZ_BIN, then topaz on PATH)"
    )
    execution.add_argument(
        "--overwrite", action="store_true", help="replace picks of tilt series that already have them"
    )
    execution.add_argument(
        "--no_save_tilts", action="store_true", help="do not keep the preprocessed tilt images Topaz picked on"
    )
    execution.add_argument("--work_dir", default=None, help="scratch directory (default: a folder under $TMPDIR)")
    execution.add_argument("--keep_work", action="store_true", help="keep each tilt series' scratch folder")
    execution.add_argument("--dry_run", action="store_true", help="list what would be processed, then exit")
    return parser


def _devices(spec: str | None) -> list[int]:
    """GPU indices as Topaz numbers them (relative to the visible GPUs); ``[-1]`` means CPU."""
    import os
    import subprocess

    if spec:
        if spec.strip().lower() in ("cpu", "-1"):
            return [-1]
        return [int(x) for x in spec.split(",") if x.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [x for x in visible.split(",") if x.strip() and x.strip() != "-1"]
        return list(range(len(ids))) or [-1]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return list(range(sum(1 for line in out.stdout.splitlines() if line.startswith("GPU ")))) or [-1]
    except (OSError, subprocess.TimeoutExpired):
        return [-1]


def main(args: argparse.Namespace) -> None:
    """Run ``jolt detect_2d``."""
    import json
    import queue
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from jolt import topaz
    from jolt.io import config
    from jolt.utils import FileLog, parse_name_list

    if args.radius <= 0:
        raise SystemExit("--radius must be a positive number of pixels")
    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("--top_k must be positive")
    threshold = args.threshold if (args.threshold is not None or args.top_k is not None) else topaz.DEFAULT_THRESHOLD

    project_dir = Path(args.project_dir)
    cfg = config.load(project_dir)
    series = cfg["tilt_series"]
    names = parse_name_list(args.ts) or sorted(series)
    unknown = [n for n in names if n not in series]
    if unknown:
        raise SystemExit(f"tilt series not in {config.config_path(project_dir)}: {unknown}")
    weights = args.weights or cfg.get("commands", {}).get("finetune", {}).get("outputs", {}).get("weights")
    model = topaz.resolve_model(weights)
    out_root = project_dir / "detect_2d"
    star_of = {n: out_root / n / f"{n}_picks.star" for n in names}
    existing = [n for n in names if star_of[n].exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{len(existing)} tilt series already have picks in {out_root} ({existing[:5]}); use --overwrite"
        )
    devices = _devices(args.devices)
    selection = f"top_k={args.top_k}" if args.top_k is not None else f"threshold={threshold}"
    print(f"detect_2d: {len(names)} tilt series, model {model}, radius {args.radius}, {selection}, devices {devices}")
    if args.dry_run:
        for n in names:
            print(f"  {series[n]['raw_stack']} -> {star_of[n]}")
        return

    install = topaz.find_topaz(args.topaz_bin)
    out_root.mkdir(parents=True, exist_ok=True)
    log = FileLog(out_root / "detect_2d.log")
    log(f"topaz: {install.executable} ({install.version}); seeded: {install.python is not None}")
    if install.python is None:
        log("WARNING: Topaz's Python interpreter was not found; preprocessing is unseeded and picks vary between runs")
    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="jolt_topaz_"))
    pool_devices: queue.Queue = queue.Queue()
    for d in devices:
        pool_devices.put(d)

    def run(name):
        device = pool_devices.get()
        try:
            stats = topaz.detect_tilt_series(
                name,
                series[name]["raw_stack"],
                int(series[name]["n_tilts"]),
                star_of[name],
                topaz=install,
                model=model,
                radius=args.radius,
                threshold=None if args.top_k is not None else threshold,
                top_k=args.top_k,
                seed=args.seed,
                device=device,
                work_dir=work_root / name,
                tilts_dir=None if args.no_save_tilts else out_root / name / "tilts",
                keep_work=args.keep_work,
                log=log,
            )
            return name, {"status": "done", **stats}
        except Exception as exc:
            log(f"[{name}] FAILED: {type(exc).__name__}: {exc} (scratch kept in {work_root / name})")
            return name, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            pool_devices.put(device)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        results = dict(executor.map(run, names))
    with open(out_root / "summary.json", "w") as handle:
        json.dump(results, handle, indent=2)
    for name, result in results.items():
        if result["status"] == "done":
            series[name]["picks_2d"] = str(star_of[name].resolve())
            series[name]["picks_frame"] = "raw"
    parameters = {k: v for k, v in vars(args).items() if k != "func"}
    parameters.update({"model": model, "threshold_used": threshold, "topaz_version": install.version})
    config.record_command(cfg, "detect_2d", parameters, {"directory": str(out_root.resolve())})
    config.save(project_dir, cfg)
    failed = [n for n, r in results.items() if r["status"] != "done"]
    detail = f": {', '.join(failed)}" if failed else ""
    log(f"{len(names) - len(failed)} tilt series picked; {len(failed)} failed{detail}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(add_args().parse_args())
