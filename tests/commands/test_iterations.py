"""joint_optimize --iter: later passes start from the previous refined alignment and improve a misaligned input."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import starfile
import torch

from jolt import pipeline
from tests.synthetic import synthetic_picks

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
    xf = np.loadtxt(d / "out" / "S_refined.xf", ndmin=2)[:, 4:6]
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
