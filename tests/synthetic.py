"""Synthetic tilt series for tests: random 3D particles projected exactly into every tilt."""

import numpy as np

from jolt.geometry import center_to_corner_2d_bottom_left, project_y_axis


def synthetic_picks(n_particles=30, tilts=None, image_hw=(256, 256), depth=60, noise_px=0.3, seed=0):
    """Return (centered 3D points, {tilt: (n, 3) picks x, y, score} in the STAR corner frame, tilts)."""
    rng = np.random.default_rng(seed)
    tilts = list(np.arange(-45.0, 46.0, 3.0)) if tilts is None else list(tilts)
    h, w = image_hw
    pts = np.column_stack(
        [
            rng.uniform(-w / 2 + 50, w / 2 - 50, n_particles),
            rng.uniform(-h / 2 + 30, h / 2 - 30, n_particles),
            rng.uniform(-depth / 2 + 8, depth / 2 - 8, n_particles),
        ]
    )
    picks = {}
    for t in tilts:
        uv = project_y_axis(pts, t) + rng.normal(0.0, noise_px, (n_particles, 2))
        corner = center_to_corner_2d_bottom_left(uv, image_hw)
        picks[float(t)] = np.column_stack([corner, rng.normal(0.0, 1.0, n_particles)])
    return pts, picks, tilts
