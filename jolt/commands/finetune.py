"""Fine-tune the Topaz 2D particle picker on known particles (optional step).

``jolt finetune`` takes known particle positions, either 3D (projected into every raw tilt image with the
project's geometry: tilt angle from the .tlt, inverse .xf, tomogram size from jolt.yaml) or 2D labels given
directly on the raw tilt images. It preprocesses the raw tilt images with Topaz (seeded), trains Topaz from a
pretrained model, keeps every epoch's model, and records in jolt.yaml the weights ``jolt detect_2d`` uses: by default
the average of the last quarter of the epochs (``--select``). jolt.yaml also lists the held-out AUPRC of every
epoch (``--heldout_ts`` series); any single epoch can be used with
``jolt detect_2d --weights <project>/finetune/train/models/model_epochNN.sav``.

Defaults follow a benchmark on three in-situ ribosome datasets (EMPIAR-10499, an E. coli dataset and CZI
DS-10301): 12,000 steps (12 x 1,000), resnet16 with 64 units, learning rate 2e-4, and the class prior set from the
labels (``--num_particles``), and the averaged weights. JOLT computes the class prior itself and passes it to Topaz
(``--pi``), because Topaz 0.3.18 overestimates it for 2D images, and replaces Topaz's positive-crop sampler with an
O(1) draw from the same distribution (stock Topaz slows down in proportion to the number of labels).

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
    training.add_argument(
        "--model",
        default="resnet16",
        help="Topaz model to start from: a pretrained name (resnet16, resnet8) or a .sav file, e.g. a released "
        "fine-tuned model",
    )
    training.add_argument(
        "--units", type=int, default=64, help="units of the pretrained model (64 or 32); ignored for a .sav --model"
    )
    training.add_argument("--radius", type=int, default=3, help="Topaz training label radius (pixels)")
    training.add_argument(
        "--num_particles",
        type=float,
        default=None,
        help="expected number of particles per tilt image, all of them, labeled or not (default: "
        "--num_particles_factor x the mean number of labels per training tilt image). Must exceed that mean, or "
        "Topaz's GE-binomial objective would treat the labels as complete",
    )
    training.add_argument(
        "--num_particles_factor",
        type=float,
        default=1.5,
        help="when --num_particles is not given, use this multiple of the mean labels per training tilt image "
        "(labels are usually incomplete)",
    )
    training.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    training.add_argument("--minibatch_size", type=int, default=256, help="minibatch size")
    training.add_argument("--epoch_size", type=int, default=1000, help="minibatches per epoch")
    training.add_argument(
        "--epochs", type=int, default=12, help="number of epochs; every epoch's model is kept (see --select)"
    )
    training.add_argument(
        "--select",
        choices=["average", "best_auprc", "last"],
        default="average",
        help="weights recorded for detect_2d: average = uniform weight average of the last quarter of the epochs "
        "(10-12 of 12; the most accurate and stable in our benchmark), best_auprc = the epoch with the best held-out "
        "AUPRC (noisy with few held-out tilt series; can pick an early, under-trained epoch), last = final epoch",
    )
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


def class_prior_settings(
    n_images: int,
    n_labels: int,
    mean_image_area: float,
    radius: int,
    num_particles: float | None = None,
    factor: float = 1.5,
) -> dict:
    """Return the expected particles per image and the GE-binomial class prior pi for the training tables.

    ``num_particles`` defaults to ``factor`` x the mean labels per training image. It must exceed that mean:
    otherwise pi does not exceed the observed label fraction and Topaz silently replaces GE-binomial (which allows
    for unlabeled particles) by PN (which treats every unlabeled pixel as background).
    """
    from jolt import topaz

    if n_images <= 0 or n_labels <= 0:
        raise SystemExit("no labeled training tilt images")
    per_image = n_labels / n_images
    n = factor * per_image if num_particles is None else float(num_particles)
    if n <= per_image:
        raise SystemExit(
            f"--num_particles {n:g} must exceed the mean labels per training tilt image ({per_image:.1f}); "
            "with fewer, Topaz would treat the labels as complete (PN). Set it to the expected number of all particles"
        )
    return {
        "num_particles": n,
        "labels_per_image": per_image,
        "pi": topaz.class_prior(n, radius, mean_image_area),
        "p_observed_estimate": topaz.class_prior(per_image, radius, mean_image_area),
        "disc_pixels": topaz.disc_pixels(radius),
    }


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
    final_model = topaz.checkpoint_path(out / "train" / "models", args.epochs, args.epochs)
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
        deselected = {int(i) - 1 for i in entry.get("deselected_tilts", [])}  # tilt images deselected in Warp
        wanted = [i for i in sorted(labels[name]) if abs(angles[i]) <= args.max_abs_angle + 0.5 and i not in deselected]
        if deselected & set(labels[name]):
            log(f"[{name}] skipping {len(deselected & set(labels[name]))} labeled tilt image(s) deselected in Warp")
        if not args.dry_run and not all(
            (tilts_dir / f"{topaz.tilt_image_name(i, n_tilts)}.mrc").exists() for i in wanted
        ):
            work = out / "work" / name
            split = topaz.split_stack(entry["raw_stack"], work)
            topaz.preprocess(install, split, tilts_dir, devices[0], args.seed, out / "topaz_preprocess.log")
            for p in split:
                p.unlink()
        role = "test" if name in heldout else "train"
        nx, ny = entry["raw_dims_xy"]
        for i in wanted:
            stem = topaz.tilt_image_name(i, n_tilts)
            images.append((f"{name}_{stem}", tilts_dir / f"{stem}.mrc", labels[name][i], role, nx * ny))
    data_dir = out / "data"
    counts = topaz.write_training_files([im[:4] for im in images], data_dir, args.radius)
    log(f"training data: {json.dumps(counts)}")
    train_areas = [im[4] for im in images if im[3] == "train"]
    prior = class_prior_settings(
        counts["train"]["n_images"],
        counts["train"]["labels_kept"],
        sum(train_areas) / max(len(train_areas), 1),
        args.radius,
        args.num_particles,
        args.num_particles_factor,
    )
    log(
        f"class prior: {prior['labels_per_image']:.1f} labels per training tilt image -> num_particles "
        f"{prior['num_particles']:.1f}, pi {prior['pi']:.5f} (label fraction ~{prior['p_observed_estimate']:.5f}; "
        f"disc {prior['disc_pixels']} px at radius {args.radius})"
    )
    if args.dry_run:
        log(f"dry run: training tables in {data_dir}")
        return

    log(f"training with Topaz {install.version} on device {devices[0]} (seeded: {install.python is not None})")
    if devices[0] < 0:
        log("WARNING: training on the CPU is slow (hours); use a CUDA Topaz (--topaz_bin) on a GPU node if possible")
    info = topaz.train(
        install,
        data_dir,
        out / "train",
        model=args.model,
        units=args.units,
        radius=args.radius,
        pi=prior["pi"],
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        epoch_size=args.epoch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=devices[0],
        select=args.select,
        log=log,
    )
    info["class_prior"] = prior
    info["topaz_version"] = install.version
    with open(out / "train_summary.json", "w") as handle:
        json.dump({"counts": counts, **info}, handle, indent=2)
    if info["model"] is None:
        raise SystemExit(f"training failed ({info.get('aborted') or 'see ' + str(out / 'train' / 'topaz_train.log')})")
    parameters = {k: v for k, v in vars(args).items() if k != "func"}
    for warning in info["warnings"]:
        if not warning.startswith("WARNING"):
            log(f"WARNING: {warning}")
    outputs = {
        "weights": info["model"],
        "selection": info["selection"],
        "selected_epochs": info["selected_epochs"],
        "best_auprc_epoch": info["best_auprc_epoch"],
        "final_weights": info["final_model"],
        "test_auprc_by_epoch": info["test_auprc_by_epoch"],
        "num_particles": prior["num_particles"],
        "pi": prior["pi"],
        "sampler": info["sampler"],
        "topaz_version": install.version,
        "directory": str(out.resolve()),
    }
    config.record_command(cfg, "finetune", parameters, outputs)
    config.save(project_dir, cfg)
    for epoch, value in enumerate(info["test_auprc_by_epoch"], start=1):
        mark = "  <- used" if epoch in info["selected_epochs"] else ""
        log(f"  epoch {epoch:3d}  held-out AUPRC {value:.4f}{mark}")
    what = (
        f"average of epochs {info['selected_epochs'][0]}-{info['selected_epochs'][-1]}"
        if len(info["selected_epochs"]) > 1
        else f"epoch {info['selected_epochs'][0]} of {args.epochs}"
    )
    log(
        f"model: {info['model']} ({what}, --select {info['selection']}; used by jolt detect_2d by default). "
        f"Every epoch is kept in {out / 'train' / 'models'}"
    )


if __name__ == "__main__":
    main(add_args().parse_args())
