"""joint_optimize --iter: later passes start from the previous refined alignment and improve a misaligned input."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import starfile
import torch

from jolt import pipeline
from tests.synthetic import synthetic_picks, synthetic_series

TOMO_XYZ = (256, 256, 60)


def _run(tmp_path, error, iterations, name):
    pts, picks, tilts = synthetic_picks(n_particles=40, image_hw=(256, 256), depth=60, noise_px=0.5, seed=3)
    d = tmp_path / name
    d.mkdir()
    rows = [(f"tilt_{i + 1:02d}", x, y, s) for i, t in enumerate(tilts) for x, y, s in picks[float(t)]]
    starfile.write(
        pd.DataFrame(
            rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
        ),
        d / "picks.star",
    )
    (d / "s.tlt").write_text("".join(f"{t:.2f}\n" for t in tilts))
    # The picks are exact; the input .xf carries a per-tilt error that joint_optimize has to remove.
    (d / "s.xf").write_text("".join(f"1.0 0.0 0.0 1.0 {dx:.3f} {dy:.3f}\n" for dx, dy in error))
    inputs = pipeline.TiltSeriesInputs("S", d / "picks.star", "raw", d / "s.xf", d / "s.tlt", (256, 256), TOMO_XYZ)
    settings = replace(pipeline.JointOptimizeSettings(), iterations=iterations)
    status = pipeline.run_tilt_series(inputs, d / "out", settings, device=torch.device("cpu"), log=lambda m: None)
    xf = np.loadtxt(d / "out" / "S.xf", ndmin=2)[:, 4:6]
    left = xf - xf.mean(axis=0)  # truth is the identity; a common translation is a gauge freedom
    return status, float(np.sqrt(np.mean(np.sum(left**2, axis=1))))


def test_iterations_improve_a_misaligned_input(tmp_path):
    rng = np.random.default_rng(1)
    error = rng.uniform(-4.0, 4.0, size=(31, 2))
    one, left1 = _run(tmp_path, error, 1, "one")
    three, left3 = _run(tmp_path, error, 3, "three")
    assert len(one["iterations"]) == 1
    assert len(three["iterations"]) == 3
    assert json.dumps(three["iterations"])  # recorded and serializable
    initial = float(np.sqrt(np.mean(np.sum((error - error.mean(axis=0)) ** 2, axis=1))))
    assert left3 < left1 < initial


def _run_aligned(tmp_path, series, true_rows, stack_rows, raw_xy, tomo_xyz, iterations):
    """Run one tilt series whose picks were made on an aligned stack built with ``stack_rows``."""
    d = tmp_path / f"aligned_{iterations}"
    d.mkdir()
    rows = [(f"tilt_{i + 1:02d}", x, y, s) for i, t in enumerate(series.tilts) for x, y, s in series.picks[float(t)]]
    starfile.write(
        pd.DataFrame(
            rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
        ),
        d / "picks.star",
    )
    (d / "s.tlt").write_text("".join(f"{t:.2f}\n" for t in series.tilts))
    (d / "s.xf").write_text("".join(" ".join(f"{v:.6f}" for v in row) + "\n" for row in stack_rows))
    inputs = pipeline.TiltSeriesInputs("S", d / "picks.star", "aligned", d / "s.xf", d / "s.tlt", raw_xy, tomo_xyz)
    settings = replace(pipeline.JointOptimizeSettings(), iterations=iterations)
    status = pipeline.run_tilt_series(inputs, d / "out", settings, device=torch.device("cpu"), log=lambda m: None)
    xf = np.loadtxt(d / "out" / "S.xf", ndmin=2)
    np.testing.assert_allclose(xf[:, :4], true_rows[:, :4], atol=1e-6)  # only the shifts are refined
    left = xf[:, 4:6] - true_rows[:, 4:6]
    left -= left.mean(axis=0)  # a common translation is a gauge freedom
    return status, float(np.sqrt(np.mean(np.sum(left**2, axis=1))))


def test_iterations_with_picks_on_the_aligned_stack(tmp_path):
    """Picks made on the aligned stack follow the refined alignment: later passes must not re-apply the correction.

    Realistic layout: an 85 deg tilt axis on a non-square detector (raw 320 x 384, aligned 384 x 320), Topaz-like
    picks (several species, clutter, missed detections), and an aligned stack built from an ``.xf`` whose shifts are
    off by up to 6 px per tilt.
    """
    tomo_xyz = (384, 320, 100)
    raw_xy = (320, 384)
    series = synthetic_series(n_particles=150, image_hw=(tomo_xyz[1], tomo_xyz[0]), depth=tomo_xyz[2], shift_px=6.0)
    rng = np.random.default_rng(7)
    a = np.deg2rad(85.0)
    rotation = [np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]
    true_rows = np.array([rotation + list(rng.uniform(-20.0, 20.0, 2)) for _ in series.tilts])
    # synthetic_series adds its shifts to the picks in the internal frame (y up); the STAR and .xf frames are y down.
    stack_rows = true_rows.copy()
    stack_rows[:, 4] += series.shifts[:, 0]
    stack_rows[:, 5] -= series.shifts[:, 1]
    injected = series.shifts - series.shifts.mean(axis=0)
    initial = float(np.sqrt(np.mean(np.sum(injected**2, axis=1))))

    _, left1 = _run_aligned(tmp_path, series, true_rows, stack_rows, raw_xy, tomo_xyz, 1)
    three, left3 = _run_aligned(tmp_path, series, true_rows, stack_rows, raw_xy, tomo_xyz, 3)
    assert len(three["iterations"]) == 3
    # Re-applying the first pass's correction makes every pass change the shifts by the same amount and leaves
    # more error after 3 passes than in the input (6.6 px for 5.2 px injected, before the fix).
    changes = [it["shift_change_rms_px"] for it in three["iterations"]]
    assert changes[2] < changes[0]
    assert left3 < left1 < initial
