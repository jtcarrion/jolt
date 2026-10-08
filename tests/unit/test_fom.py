"""Figure of merit: scored against the refined alignment, in the picks' own frame."""

import numpy as np
import torch

from jolt import fom
from jolt.geometry import center_to_corner_2d_bottom_left, corner_to_center_3d, project_points, tilt_projection_matrices

TOMO_ZYX = (80, 200, 240)
IMAGE_HW = (200, 240)
TILTS = [-40.0, -20.0, 0.0, 20.0, 40.0]


def _points(n=12, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = TOMO_ZYX
    # Spread in y on purpose: mirroring y would move these points far away from their picks.
    return np.column_stack([rng.uniform(30, x - 30, n), rng.uniform(20, y - 60, n), rng.uniform(15, z - 15, n)])


def _picks(points_corner, shifts_by_tilt):
    """STAR-frame picks (y down) at each point's projection plus the given internal-frame shift."""
    center = torch.as_tensor(corner_to_center_3d(points_corner, TOMO_ZYX), dtype=torch.float32)
    proj = project_points(center, tilt_projection_matrices(TILTS)).numpy()
    coords = {}
    for v, tilt in enumerate(TILTS):
        uv = proj[:, v, :].astype(float) + np.asarray(shifts_by_tilt.get(tilt, (0.0, 0.0)))
        xy = center_to_corner_2d_bottom_left(uv, IMAGE_HW)
        coords[tilt] = np.column_stack([xy, np.ones(len(xy))]).astype(np.float32)
    return coords


def test_residual_uses_the_refined_shifts():
    points = _points()
    shifts = {t: (3.0 * np.sign(t + 0.1), -2.0) for t in TILTS}
    coords = _picks(points, shifts)
    without = fom.reprojection_residuals(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, torch.device("cpu"))
    with_shifts = fom.reprojection_residuals(
        points, coords, TILTS, IMAGE_HW, TOMO_ZYX, torch.device("cpu"), shifts_by_tilt=shifts
    )
    np.testing.assert_allclose(without, np.sqrt(13.0), atol=1e-3)
    np.testing.assert_allclose(with_shifts, 0.0, atol=1e-3)


def test_benchmark_features_match_picks_in_their_own_frame():
    points = _points()
    shifts = {t: (4.0, 3.0) for t in TILTS}
    coords = _picks(points, shifts)
    good = fom._benchmark_features(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, shifts)
    np.testing.assert_allclose(good["match_fraction"], 1.0)
    np.testing.assert_allclose(good["global_coverage"], 1.0)
    # Without the shifts every projection is 5 px off, which is not a "good" (< 5 px) match.
    off = fom._benchmark_features(points, coords, TILTS, IMAGE_HW, TOMO_ZYX)
    assert np.all(off["global_coverage"] < 1.0)


def test_figure_of_merit_is_high_once_the_shifts_are_applied():
    points = _points()
    shifts = {t: (6.0, -6.0) for t in TILTS}
    coords = _picks(points, shifts)
    kwargs = dict(search_radius=10.0, device=torch.device("cpu"))
    refined = fom.figure_of_merit(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, shifts_by_tilt=shifts, **kwargs)
    unrefined = fom.figure_of_merit(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, **kwargs)
    assert np.all(refined["fom"] > 0.97)  # two benchmark terms are ranked against the other points
    assert np.all(unrefined["fom"] < 0.3)


def test_matched_residual_ignores_tilts_without_a_nearby_pick():
    points = _points(n=1, seed=2)  # one particle, so a moved pick cannot land near another particle
    coords = _picks(points, {})
    exact = fom.matched_residuals(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, 12.0, torch.device("cpu"))
    # Move every pick of one tilt far away: the all-tilt RMS grows, the matched residual ignores that tilt.
    far = {
        t: (c.copy() if t != TILTS[0] else c + np.array([60.0, 0.0, 0.0], dtype=np.float32)) for t, c in coords.items()
    }
    all_rms = fom.reprojection_residuals(points, far, TILTS, IMAGE_HW, TOMO_ZYX, torch.device("cpu"))
    res, n = fom.matched_residuals(points, far, TILTS, IMAGE_HW, TOMO_ZYX, 12.0, torch.device("cpu"))
    np.testing.assert_allclose(exact[0], 0.0, atol=1e-3)
    np.testing.assert_array_equal(exact[1], len(TILTS))
    np.testing.assert_allclose(all_rms, np.sqrt(60.0**2 / len(TILTS)), rtol=1e-3)  # the moved tilt counts here
    np.testing.assert_allclose(res, 0.0, atol=1e-3)
    np.testing.assert_array_equal(n, len(TILTS) - 1)


def test_matched_residual_is_infinite_without_any_matched_tilt():
    points = _points(n=3, seed=4)
    coords = {t: c + np.array([80.0, 80.0, 0.0], dtype=np.float32) for t, c in _picks(points, {}).items()}
    res, n = fom.matched_residuals(points, coords, TILTS, IMAGE_HW, TOMO_ZYX, 12.0, torch.device("cpu"))
    assert np.all(np.isinf(res)) and np.all(n == 0)
