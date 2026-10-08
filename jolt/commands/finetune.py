"""Fine-tune the Topaz 2D particle picker on known particles (optional step).

``jolt finetune`` takes known particle positions, either 3D (projected into every raw tilt image with the
project's geometry: tilt angle from the .tlt, inverse .xf, tomogram size from jolt.yaml) or 2D labels given
directly on the raw tilt images. It preprocesses the raw tilt images with Topaz (seeded), trains Topaz from a
pretrained model, and records the final model in jolt.yaml so that ``jolt detect_2d`` uses it.

Label files are STAR files: either one file for all tilt series, with the tilt series in ``rlnTomoName`` or
``rlnMicrographName`` (``TS_001`` or ``TS_001.tomostar``), or a path pattern in which ``{ts}`` is replaced by
each tilt-series name. 3D labels are ``rlnCoordinateX/Y/Z`` in tomogram pixels at the tilt-stack pixel size,
in the frame of the ``jolt joint_optimize`` particle files. 2D labels are ``rlnCoordinateX/Y`` on the raw tilt
images, with the tilt image named by its stack index in ``rlnMicrographName`` (``tilt_07``, ``TS_001_07``).
"""

import argparse

from jolt.utils import HelpFormatter


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt finetune`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)

    parser.add_argument("project_dir", help="JOLT project directory (created by jolt import)")

    labels = parser.add_argument_group("Labels (exactly one)")
    source = labels.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels_3d",
        default=None,
        help="known 3D particle positions (tomogram voxels at the tilt-stack pixel size, 0-based, in the frame "
        "given by --labels_frame); projected into every raw tilt image to make 2D labels",
    )
    source.add_argument("--labels_2d", default=None, help="2D labels on the raw tilt images (pixels), used as they are")
    labels.add_argument(
        "--labels_frame",
        choices=["warp", "teamtomo"],
        default="warp",
        help="coordinate frame of --labels_3d: warp (WarpTools tomograms, e.g. particles picked or refined on Warp "
        "reconstructions; sub-voxel centers from each tilt series' Warp XML) or teamtomo (TeamTomo, IMOD clip rotx "
        "and CZI portal tomograms). Same meaning as joint_optimize --output_frame",
    )

    training = parser.add_argument_group("Training")
    training.add_argument("--heldout_ts", required=True, help="comma-separated tilt series held out for validation")
    training.add_argument("--exclude_ts", default=None, help="comma-separated tilt series left out entirely")
    training.add_argument(
        "--max_abs_angle",
        type=float,
        default=40.0,
        help="use only tilt images whose absolute tilt angle is <= this (degrees)",
    )
    training.add_argument("--model", default="resnet16", help="Topaz model to start from")
    training.add_argument("--radius", type=int, default=3, help="Topaz training label radius (pixels)")
    training.add_argument(
        "--num_particles", type=float, default=350, help="expected number of particles per tilt image (all of them)"
    )
    training.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    training.add_argument("--minibatch_size", type=int, default=256, help="minibatch size")
    training.add_argument("--epoch_size", type=int, default=100, help="minibatches per epoch")
    training.add_argument("--epochs", type=int, default=20, help="number of epochs; the final epoch's model is used")
    training.add_argument("--seed", type=int, default=0, help="random seed for Topaz (numpy, torch, random)")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--devices", default=None, help="GPU index to train on, e.g. 0, or 'cpu' (default: first)")
    execution.add_argument(
        "--topaz_bin", default=None, help="Topaz executable (default: $JOLT_TOPAZ_BIN, then topaz on PATH)"
    )
    execution.add_argument("--overwrite", action="store_true", help="retrain even if a trained model exists")
    execution.add_argument("--dry_run", action="store_true", help="build and check the training data only")
    return parser


def _label_tables(spec: str, names: list[str]):
    """Return ``{tilt series: table}`` from a ``{ts}`` pattern or a single STAR file with a tilt-series column."""
    from pathlib import Path

    from jolt.io import read_star_table

    if "{ts}" in spec:
        tables = {}
        for name in names:
            path = Path(spec.replace("{ts}", name))
            if path.is_file():
                tables[name] = read_star_table(path)
        return tables
    table = read_star_table(spec)
    column = next((c for c in ("rlnTomoName", "rlnMicrographName") if c in table.columns), None)
    if column is None:
        raise SystemExit(f"{spec}: needs rlnTomoName or rlnMicrographName to tell tilt series apart (or use {{ts}})")
    keys = table[column].astype(str).str.replace(r"\.tomostar$", "", regex=True)
    return {name: table[keys == name].reset_index(drop=True) for name in names if (keys == name).any()}


def labels_from_3d(table, entry: dict, tomo_dims_xyz, frame=None):
    """Project 3D labels into every raw tilt image; returns ``{0-based stack index: (n, 2) raw pixels}``.

    ``frame`` (a :class:`jolt.geometry.OutputFrame`) converts the labels from their frame to JOLT's own frame first;
    ``None`` means they are already in JOLT's frame.
    """
    from jolt.geometry import project_star_points_to_raw
    from jolt.io import read_tlt, read_xf

    tilts = read_tlt(entry["tlt"])
    rows = read_xf(entry["xf"], tilts)
    xyz = table[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=float)
    if frame is not None:
        xyz = frame.from_frame(xyz)
    nx, ny = entry["raw_dims_xy"]
    out = {}
    for i, (tilt, row) in enumerate(zip(tilts, rows, strict=True)):
        xy = project_star_points_to_raw(xyz, tilt, row, tomo_dims_xyz, (nx, ny))
        inside = (xy[:, 0] >= 0) & (xy[:, 0] <= nx - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= ny - 1)
        out[i] = xy[inside]
    return out


def labels_from_2d(table, n_tilts: int):
    """Group 2D labels by tilt image; returns ``{0-based stack index: (n, 2) raw pixels}``."""
    import numpy as np

    from jolt.pipeline import tilt_index_from_name

    out: dict[int, list] = {}
    keep = np.ones(len(table), dtype=bool)
    if "rlnAutopickFigureOfMerit" in table.columns:
        keep = table["rlnAutopickFigureOfMerit"].to_numpy(dtype=float) != -999.0  # rows rejected in label editors
    for name, x, y, ok in zip(
        table["rlnMicrographName"].astype(str), table["rlnCoordinateX"], table["rlnCoordinateY"], keep, strict=True
    ):
        if not ok:
            continue
        idx = tilt_index_from_name(name) - 1
        if not 0 <= idx < n_tilts:
            raise SystemExit(f"2D label on tilt image {name!r} is outside the stack (1..{n_tilts})")
        out.setdefault(idx, []).append((float(x), float(y)))
    return {i: np.array(v, dtype=np.float64) for i, v in out.items()}


def main(args: argparse.Namespace) -> None:
    """Run ``jolt finetune``."""
    import json
    from pathlib import Path

    from jolt import topaz
    from jolt.commands.detect_2d import _devices
    from jolt.io import config, read_tlt
    from jolt.pipeline import output_frame_for
    from jolt.utils import FileLog, parse_name_list

    project_dir = Path(args.project_dir)
    cfg = config.load(project_dir)
    series = cfg["tilt_series"]
    heldout = set(parse_name_list(args.heldout_ts) or [])
    exclude = set(parse_name_list(args.exclude_ts) or [])
    names = [n for n in sorted(series) if n not in exclude]
    unknown = sorted((heldout | exclude) - set(series))
    if unknown:
        raise SystemExit(f"tilt series not in {config.config_path(project_dir)}: {unknown}")
    out = project_dir / "finetune"
    final_model = out / "train" / "models" / f"model_epoch{args.epochs}.sav"
    if final_model.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"{final_model} exists; use --overwrite to retrain")
    out.mkdir(parents=True, exist_ok=True)
    log = FileLog(out / "finetune.log")
    tomo_dims = tuple(cfg["project"]["tomo_dims_xyz"])
    tables = _label_tables(args.labels_3d or args.labels_2d, names)
    missing_heldout = sorted(heldout - set(tables))
    if missing_heldout:
        raise SystemExit(f"no labels for held-out tilt series {missing_heldout}")

    labels: dict[str, dict] = {}
    for name in names:
        if name not in tables:
            log(f"[{name}] no labels; not used")
            continue
        entry = series[name]
        if args.labels_3d:
            frame = output_frame_for(
                args.labels_frame, tomo_dims, entry["pixel_size"], entry["xf"], entry["tlt"], entry.get("xml")
            )
            labels[name] = labels_from_3d(tables[name], entry, tomo_dims, frame)
            if "exact binning" in frame.source and args.labels_frame == "warp":
                log(f"[{name}] WARNING: Warp XML has no physical dimensions; warp labels assume exact binning")
        else:
            labels[name] = labels_from_2d(tables[name], int(entry["n_tilts"]))
        rows = [
            (topaz.tilt_image_name(i, int(entry["n_tilts"])), f"{x:.2f}", f"{y:.2f}", 1.0, name)
            for i in sorted(labels[name])
            for x, y in labels[name][i]
        ]
        (out / "labels").mkdir(exist_ok=True)
        topaz.write_picks_star(out / "labels" / f"{name}_labels.star", rows)
        log(
            f"[{name}] {len(rows)} 2D labels on {len(labels[name])} tilt images "
            f"({'held out' if name in heldout else 'training'})"
        )
    if not set(labels) - heldout:
        raise SystemExit("no training tilt series with labels (all are held out or unlabeled)")

    devices = _devices(args.devices)
    install = None if args.dry_run else topaz.find_topaz(args.topaz_bin)
    images = []
    for name in labels:
        entry = series[name]
        n_tilts = int(entry["n_tilts"])
        angles = read_tlt(entry["tlt"])
        tilts_dir = out / "tilts" / name
        wanted = [i for i in sorted(labels[name]) if abs(angles[i]) <= args.max_abs_angle + 0.5]
        if not args.dry_run and not all(
            (tilts_dir / f"{topaz.tilt_image_name(i, n_tilts)}.mrc").exists() for i in wanted
        ):
            work = out / "work" / name
            split = topaz.split_stack(entry["raw_stack"], work)
            topaz.preprocess(install, split, tilts_dir, devices[0], args.seed, out / "topaz_preprocess.log")
            for p in split:
                p.unlink()
        role = "test" if name in heldout else "train"
        for i in wanted:
            stem = topaz.tilt_image_name(i, n_tilts)
            images.append((f"{name}_{stem}", tilts_dir / f"{stem}.mrc", labels[name][i], role))
    data_dir = out / "data"
    counts = topaz.write_training_files(images, data_dir, args.radius)
    log(f"training data: {json.dumps(counts)}")
    if args.dry_run:
        log(f"dry run: training tables in {data_dir}")
        return

    log(f"training with Topaz {install.version} on device {devices[0]} (seeded: {install.python is not None})")
    info = topaz.train(
        install,
        data_dir,
        out / "train",
        model=args.model,
        radius=args.radius,
        num_particles=args.num_particles,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        epoch_size=args.epoch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=devices[0],
        log=log,
    )
    with open(out / "train_summary.json", "w") as handle:
        json.dump({"counts": counts, **info}, handle, indent=2)
    if info["model"] is None:
        raise SystemExit(f"training failed ({info.get('aborted') or 'see ' + str(out / 'train' / 'topaz_train.log')})")
    parameters = {k: v for k, v in vars(args).items() if k != "func"}
    config.record_command(cfg, "finetune", parameters, {"weights": info["model"], "directory": str(out.resolve())})
    config.save(project_dir, cfg)
    log(
        f"model: {info['model']} (used by jolt detect_2d by default); held-out AUPRC by epoch "
        f"{info.get('test_auprc_by_epoch', [])}"
    )


if __name__ == "__main__":
    main(add_args().parse_args())
