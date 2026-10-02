"""Process one tilt series end to end for ``jolt joint_optimize``.

Stages (all in the aligned frame of the tilt series):

1. Read the 2D picks and, for picks made on the raw stack, map them into the aligned frame with the ``.xf``.
2. MLESAC: 3D hypotheses from the picks (:mod:`jolt.mlesac`).
3. Keep hypotheses inside the tomogram (10 px margin in x and y, 5 px in z).
4. Greedy bundle adjustment of the points and per-tilt shifts (:mod:`jolt.bundle_adjustment`). A failed or
   non-finite result falls back to the MLESAC points, loudly; a diverged one is kept and flagged.
5. Write the refined ``.xf``: the input row plus the bundle-adjustment shift change, in the correction
   convention IMOD's ``newstack -xf`` applies.
6. Selection: reprojection residual, border padding and near-duplicate removal (:mod:`jolt.selection`).
7. Figure of merit and mean detector score (:mod:`jolt.fom`), then the particle STAR, an IMOD ``.mod`` and
   ``run_status.json``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jolt import bundle_adjustment, fom, geometry, mlesac, selection
from jolt.io import read_star_table, read_tlt, read_xf, write_star, write_xf
from jolt.io.imod import write_points_mod

#: File-name suffixes stripped before reading the tilt index from a tilt-image name.
_IMAGE_SUFFIXES = (".mrc", ".mrcs", ".tif", ".tiff", ".eer", ".gain", ".dm4", ".star")


@dataclass
class JointOptimizeSettings:
    """Every parameter of one ``joint_optimize`` run."""

    mlesac: mlesac.MlesacSettings = field(default_factory=mlesac.MlesacSettings)
    bundle: bundle_adjustment.BundleAdjustmentSettings = field(
        default_factory=bundle_adjustment.BundleAdjustmentSettings
    )
    residual_threshold: float = 12.0
    dedup_radius: float = 5.0
    border_padding: float = 6.0
    fom_search_radius: float = 10.0
    fom_affinity_temp: float = 1.0
    fom_tau: float = 0.2
    fom_rmse_scale: float = 5.9999
    divergence_rms_px: float = 8.165
    volume_margin_xyz: tuple[float, float, float] = (10.0, 10.0, 5.0)


@dataclass
class TiltSeriesInputs:
    """Files and dimensions of one tilt series (from ``jolt.yaml``)."""

    name: str
    picks_2d: Path
    picks_frame: str  # "raw" or "aligned"
    xf: Path
    tlt: Path
    raw_dims_xy: tuple[int, int]
    tomo_dims_xyz: tuple[int, int, int]


def tilt_index_from_name(name: str) -> int:
    """Return the 1-based stack index encoded in a tilt-image name.

    The index is the integer after ``tilt_`` (``..._tilt_07_...``) or, failing that, the integer after the
    last underscore (``TS_5_07.mrc`` -> 7). A known image suffix (``.mrc``, ``.tif`` ...) is ignored.

    Raises
    ------
    ValueError
        If the name holds no such integer.
    """
    base = str(name).split("/")[-1].strip()
    stem = base
    for suffix in _IMAGE_SUFFIXES:
        if base.lower().endswith(suffix):
            stem = base[: -len(suffix)]
            break
    prefix = None
    m = re.search(r"(^|_)tilt_([+-]?\d+\.\d+)(?:_|$)", stem, flags=re.IGNORECASE)
    if m:
        prefix = m.group(2)
    else:
        m = re.search(r"(^|_)tilt_([+-]?\d+)_(\d{2})(?:_|$)", stem, flags=re.IGNORECASE)
        if m:
            prefix = f"{m.group(2)}.{m.group(3)}"
        else:
            m = re.search(r"(^|_)tilt_([+-]?\d+(?:\.\d+)?)(?:_|$)", stem, flags=re.IGNORECASE)
            if m:
                prefix = m.group(2)
    trailing = stem.split("_")[-1] if "_" in stem else stem
    if prefix is not None and re.fullmatch(r"\d+", prefix):
        return int(prefix)
    if re.fullmatch(r"\d+", trailing):
        return int(trailing)
    raise ValueError(f"cannot read a tilt index from tilt-image name {name!r} (expected e.g. TS_01 or tilt_01.mrc)")


def load_picks(
    star_path: str | PathLike,
    tilt_angles: list[float],
    picks_frame: str,
    xf_rows: list[dict],
    raw_dims_xy: tuple[int, int],
    aligned_dims_xy: tuple[int, int],
) -> dict[float, np.ndarray]:
    """Read 2D picks into ``{tilt angle: (n, 3) array of x, y, score}`` in the aligned frame.

    Tilt images are identified by their stack index (see :func:`tilt_index_from_name`), and the angle is the
    ``.tlt`` entry at that index. The dictionary follows the order in which tilt images first appear in the
    STAR file. Picks made on the raw stack are mapped with :func:`jolt.geometry.raw_picks_to_aligned`.
    """
    table = read_star_table(star_path, required=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"])
    if "rlnAutopickFigureOfMerit" in table.columns:
        scores = table["rlnAutopickFigureOfMerit"].to_numpy(dtype=float)
    else:
        scores = np.full(len(table), 0.99, dtype=float)
    xy = table[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(dtype=float)
    groups: dict[float, list[int]] = {}
    index_cache: dict[str, float] = {}
    for row, name in enumerate(table["rlnMicrographName"].astype(str)):
        tilt = index_cache.get(name)
        if tilt is None:
            idx = tilt_index_from_name(name)
            if not 1 <= idx <= len(tilt_angles):
                raise ValueError(f"{star_path}: tilt index {idx} of {name!r} is outside 1..{len(tilt_angles)}")
            tilt = float(tilt_angles[idx - 1])
            index_cache[name] = tilt
        groups.setdefault(tilt, []).append(row)
    picks = {}
    for tilt, rows in groups.items():
        arr = np.column_stack([xy[rows, 0], xy[rows, 1], scores[rows]])
        if picks_frame == "raw":
            xf_row = xf_rows[[float(r["TILT"]) for r in xf_rows].index(tilt)]
            arr = geometry.raw_picks_to_aligned(arr, xf_row, raw_dims_xy, aligned_dims_xy)
        picks[tilt] = arr
    return picks


def run_tilt_series(
    inputs: TiltSeriesInputs,
    output_dir: str | PathLike,
    settings: JointOptimizeSettings | None = None,
    device: torch.device | None = None,
    log=print,
) -> dict:
    """Run ``joint_optimize`` on one tilt series and write its outputs.

    Outputs in ``output_dir``: ``<name>_particles.star``, ``<name>_particles.mod``, ``<name>_refined.xf``
    (only when bundle adjustment succeeded) and ``run_status.json``.

    Returns
    -------
    dict
        The content of ``run_status.json``.
    """
    s = settings or JointOptimizeSettings()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    x_dim, y_dim, z_dim = (int(v) for v in inputs.tomo_dims_xyz)
    tomo_zyx = (z_dim, y_dim, x_dim)
    image_hw = (y_dim, x_dim)
    status: dict = {"tilt_series": inputs.name, "device": str(device)}

    tilts_file = [float(t) for t in read_tlt(inputs.tlt)]
    disk_rows = read_xf(inputs.xf, tilts_file)
    coords = load_picks(inputs.picks_2d, tilts_file, inputs.picks_frame, disk_rows, inputs.raw_dims_xy, (x_dim, y_dim))
    n_picks = int(sum(len(v) for v in coords.values()))
    log(f"[{inputs.name}] {n_picks} picks on {len(coords)} tilt images ({inputs.picks_frame} frame)")
    status["n_picks"] = n_picks

    result = mlesac.run_mlesac(coords, image_hw, s.mlesac, device=device, log=log)
    status["mlesac"] = {"n_points": int(len(result.X_3d)), "stage_counts": result.stage_counts}

    x_init = geometry.center_to_corner_3d(result.X_3d, tomo_zyx) if len(result.X_3d) else np.empty((0, 3))
    inside = selection.inside_volume_mask(x_init, tomo_zyx, s.volume_margin_xyz) if len(x_init) else np.zeros(0, bool)
    x_init = x_init[inside]
    status["n_points_inside_volume"] = int(len(x_init))
    log(f"[{inputs.name}] {len(x_init)} of {len(result.X_3d)} MLESAC points inside the tomogram")

    ba_tilts = sorted(coords.keys())
    shifts0 = np.zeros((len(ba_tilts), 2), dtype=np.float32)
    ba_status: dict = {"status": "not_run", "error": None}
    x_ba = shifts = None
    if len(x_init) == 0:
        ba_status = {"status": "skipped", "error": "no 3D points to refine"}
    else:
        try:
            ba = bundle_adjustment.greedy_bundle_adjustment(
                x_init, coords, ba_tilts, shifts0, image_hw, tomo_zyx, s.bundle, device=device, log=log
            )
            x_ba, shifts = ba["X"], ba["translations"]
            disp = np.linalg.norm(np.asarray(x_ba, dtype=float) - x_init, axis=1)
            rms_disp = float(np.sqrt(np.mean(disp**2))) if disp.size else 0.0
            ba_status = {
                "status": "ok",
                "error": None,
                "final_rmse_px": float(ba["rmse"]),
                "point_reg_loss": float(ba["point_reg_loss"]),
                "rms_point_displacement_px": rms_disp,
                "divergence_threshold_rms_px": float(s.divergence_rms_px),
                "match_statistics": ba["match_statistics"],
            }
            if not np.isfinite(rms_disp) or rms_disp > s.divergence_rms_px:
                ba_status["status"] = "diverged"
                log(
                    f"[{inputs.name}] WARNING: bundle adjustment diverged (RMS point displacement {rms_disp:.3f} px "
                    f"> {s.divergence_rms_px:.3f} px); its result is kept, treat this tilt series with caution"
                )
            problems = []
            if not np.isfinite(float(ba["rmse"])):
                problems.append(f"final RMSE = {ba['rmse']}")
            if not np.all(np.isfinite(np.asarray(x_ba, dtype=float))):
                problems.append("non-finite 3D points")
            if not np.all(np.isfinite(np.asarray(shifts, dtype=float))):
                problems.append("non-finite shifts")
            if problems:
                ba_status["status"] = "failed"
                ba_status["error"] = "non-finite bundle-adjustment result: " + ", ".join(problems)
                x_ba = shifts = None
        except Exception as exc:  # reported, never silent
            ba_status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            x_ba = shifts = None
        if x_ba is None:
            log(
                f"[{inputs.name}] WARNING: bundle adjustment {ba_status['status']}: {ba_status['error']}. "
                "The particles are the MLESAC points and no refined .xf is written."
            )
    ba_status["output_points_source"] = "bundle_adjustment" if x_ba is not None else "mlesac"
    ba_status["refined_xf_written"] = x_ba is not None
    status["ba"] = ba_status

    if x_ba is not None:
        delta = np.asarray(shifts, dtype=np.float64) - np.asarray(shifts0, dtype=np.float64)
        refined_rows = []
        for row in disk_rows:
            new = dict(row)
            j = next((i for i, t in enumerate(ba_tilts) if abs(float(t) - float(row["TILT"])) < 0.01), None)
            if j is not None:
                # Correction convention: DX changes sign; DY keeps it (the internal frame is y up).
                new["DX"] = float(float(row["DX"]) + (-1.0) * delta[j, 0])
                new["DY"] = float(float(row["DY"]) + (+1.0) * delta[j, 1])
            refined_rows.append(new)
        write_xf(out / f"{inputs.name}_refined.xf", refined_rows)
        status["xf_shift_change_px"] = {
            "dx": [float(r["DX"]) - float(d["DX"]) for r, d in zip(refined_rows, disk_rows, strict=True)],
            "dy": [float(r["DY"]) - float(d["DY"]) for r, d in zip(refined_rows, disk_rows, strict=True)],
        }

    points = np.asarray(x_ba if x_ba is not None else x_init, dtype=float)
    residuals = (
        fom.reprojection_residuals(points, coords, tilts_file, image_hw, tomo_zyx, device)
        if len(points)
        else np.zeros(0)
    )
    kept = (
        selection.residual_selection(
            points, residuals, s.residual_threshold, s.dedup_radius, s.border_padding, tomo_zyx
        )
        if len(points)
        else np.zeros(0, dtype=np.int64)
    )
    kept_points = points[kept]
    status["selection"] = {
        "n_points": int(len(points)),
        "n_kept": int(len(kept)),
        "residual_threshold_px": s.residual_threshold,
        "dedup_radius_px": s.dedup_radius,
        "border_padding_px": s.border_padding,
    }
    log(f"[{inputs.name}] selection kept {len(kept)} of {len(points)} points")

    try:
        scores = fom.figure_of_merit(
            kept_points,
            coords,
            tilts_file,
            image_hw,
            tomo_zyx,
            search_radius=s.fom_search_radius,
            affinity_temp=s.fom_affinity_temp,
            tau=s.fom_tau,
            rmse_scale=s.fom_rmse_scale,
            device=device,
        )
        if len(scores["fom"]) != len(kept_points):
            raise RuntimeError(f"{len(scores['fom'])} FOM values for {len(kept_points)} points")
        status["fom"] = {"status": "ok", "error": None}
    except Exception as exc:  # reported, never silent
        log(f"[{inputs.name}] WARNING: figure of merit failed ({type(exc).__name__}: {exc}); writing 1/(1+residual)")
        status["fom"] = {"status": "fallback_residual", "error": f"{type(exc).__name__}: {exc}"}
        res_kept = residuals[kept] if len(kept) else np.zeros(0)
        scores = {"fom": 1.0 / (1.0 + np.clip(res_kept, 0.0, np.inf)), "components": {}}
    fom_values = np.clip(np.nan_to_num(scores["fom"], nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    detector = scores["components"].get("cnn_raw", np.full(len(kept_points), np.nan))

    # The particle STAR is written in three steps, as the original code did, so that the six-decimal rounding
    # (and the parser's reading of it) is identical: internal frame, then y flipped to the STAR frame (y
    # down), then the score columns.
    star_path = out / f"{inputs.name}_particles.star"
    res_kept = residuals[kept] if len(kept) else np.zeros(0)
    write_star(
        star_path,
        pd.DataFrame(
            {
                "rlnCoordinateX": kept_points[:, 0],
                "rlnCoordinateY": kept_points[:, 1],
                "rlnCoordinateZ": kept_points[:, 2],
                "rlnMicrographName": f"{inputs.name}.tomostar",
                "rlnAutopickFigureOfMerit": 1.0 / (1.0 + res_kept),
            }
        ),
    )
    table = read_star_table(star_path)
    table["rlnCoordinateY"] = float(y_dim - 1) - table["rlnCoordinateY"].astype(float)
    write_star(star_path, table)
    table = read_star_table(star_path)
    table["rlnAutopickFigureOfMerit"] = fom_values
    table["rlnJoltDetectorScore"] = np.asarray(detector, dtype=float)
    table["rlnJoltResidual"] = res_kept
    write_star(star_path, table)
    try:
        write_points_mod(
            out / f"{inputs.name}_particles.mod",
            table[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(dtype=float),
        )
    except ImportError as exc:  # imodmodel is a dependency; reported if an install lacks it
        log(f"[{inputs.name}] WARNING: IMOD model not written ({exc})")
        status["mod_written"] = False
    status["n_particles"] = int(len(table))
    status["seconds"] = round(time.time() - t_start, 1)
    status["settings"] = asdict(s)
    with open(out / "run_status.json", "w") as handle:
        json.dump(status, handle, indent=2, default=float)
    log(f"[{inputs.name}] {len(table)} particles written to {out} ({status['seconds']} s)")
    return status


def run_tilt_series_logged(inputs: TiltSeriesInputs, output_dir: str, settings: JointOptimizeSettings) -> dict:
    """Run :func:`run_tilt_series` in a worker process, logging to ``<output_dir>/joint_optimize.log``.

    Returns a short summary (status of each stage and the particle count).
    """
    from jolt.utils import FileLog

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "joint_optimize.log"
    if log_path.exists():
        log_path.unlink()
    status = run_tilt_series(inputs, out, settings, log=FileLog(log_path, echo=False))
    return {
        "ba_status": status["ba"]["status"],
        "ba_error": status["ba"].get("error"),
        "fom_status": status.get("fom", {}).get("status"),
        "n_particles": status["n_particles"],
        "seconds": status["seconds"],
    }
