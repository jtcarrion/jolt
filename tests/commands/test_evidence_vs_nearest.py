"""The evidence loss against the nearest loss on a realistic synthetic series (CPU): same MLESAC start, same
selection; recall and precision at least as good, and the result stable under a 0.001 px perturbation."""

import numpy as np
import pytest
import torch

from jolt import bundle_adjustment, fom, geometry, mlesac, selection
from jolt.pipeline import JointOptimizeSettings
from tests.synthetic import CZI_LIKE_SPECIES, exclusive_match_score, synthetic_series

IMAGE_HW = (384, 384)
DEPTH = 100
TOMO_ZYX = (DEPTH, IMAGE_HW[0], IMAGE_HW[1])
RADII = {s.name: s.score_radius_px for s in CZI_LIKE_SPECIES}


@pytest.fixture(scope="module")
def series_and_start():
    series = synthetic_series(n_particles=150, image_hw=IMAGE_HW, depth=DEPTH, shift_px=3.0, seed=7)
    settings = JointOptimizeSettings()
    coords = {t: np.asarray(p, dtype=float) for t, p in series.picks.items()}
    result = mlesac.run_mlesac(coords, IMAGE_HW, settings.mlesac, device=torch.device("cpu"), log=lambda *a: None)
    x_init = geometry.center_to_corner_3d(result.X_3d, TOMO_ZYX)
    x_init = x_init[selection.inside_volume_mask(x_init, TOMO_ZYX, settings.volume_margin_xyz)]
    assert len(x_init) >= 80, "MLESAC found too few hypotheses for the comparison"
    return series, coords, x_init, settings


def _refine_and_select(loss, x_start, coords, series, settings):
    tilts = sorted(coords)
    bundle = bundle_adjustment.BundleAdjustmentSettings(loss=loss)
    if loss == "evidence":
        estimate = mlesac.estimate_noise_model(
            geometry.corner_to_center_3d(x_start, TOMO_ZYX), coords, IMAGE_HW, sigma_fallback=2.5, gamma_fallback=0.7
        )
        bundle.sigma_px = estimate.sigma_px
    ba = bundle_adjustment.greedy_bundle_adjustment(
        x_start,
        coords,
        tilts,
        np.zeros((len(tilts), 2), np.float32),
        IMAGE_HW,
        TOMO_ZYX,
        bundle,
        device=torch.device("cpu"),
        log=lambda *a: None,
    )
    refined = {float(t): (float(d[0]), float(d[1])) for t, d in zip(tilts, ba["translations"], strict=True)}
    residuals, _ = fom.matched_residuals(
        ba["X"], coords, tilts, IMAGE_HW, TOMO_ZYX, settings.bundle.max_match_distance, torch.device("cpu"), refined
    )
    kept = selection.residual_selection(
        ba["X"], residuals, None, settings.dedup_radius, settings.border_padding, TOMO_ZYX
    )
    score = exclusive_match_score(ba["X"][kept], series.xyz_corner, series.species, RADII)
    shift_err = ba["translations"] - series.shifts
    shift_err -= shift_err.mean(axis=0)
    return ba, score, float(np.sqrt(np.mean(shift_err**2)))


def test_evidence_matches_or_beats_nearest(series_and_start):
    series, coords, x_init, settings = series_and_start
    ba_n, (tp_n, fp_n, _, recall_n, precision_n), shift_n = _refine_and_select(
        "nearest", x_init, coords, series, settings
    )
    ba_e, (tp_e, fp_e, _, recall_e, precision_e), shift_e = _refine_and_select(
        "evidence", x_init, coords, series, settings
    )
    assert ba_e["loss"] == "evidence" and ba_n["loss"] == "nearest"
    assert recall_e >= recall_n - 0.02, f"recall evidence {recall_e:.3f} vs nearest {recall_n:.3f}"
    assert precision_e >= precision_n - 0.02, f"precision evidence {precision_e:.3f} vs nearest {precision_n:.3f}"
    assert shift_e <= shift_n + 0.1, f"shift error evidence {shift_e:.2f} vs nearest {shift_n:.2f} px"
    assert recall_e > 0.5 and shift_e < 1.0


def test_evidence_is_stable_under_tiny_perturbation(series_and_start):
    series, coords, x_init, settings = series_and_start
    rng = np.random.default_rng(0)
    ba_a, *_ = _refine_and_select("evidence", x_init, coords, series, settings)
    ba_b, *_ = _refine_and_select("evidence", x_init + rng.normal(0.0, 0.001, x_init.shape), coords, series, settings)
    assert np.abs(ba_a["translations"] - ba_b["translations"]).max() < 0.05
    assert np.linalg.norm(ba_a["X"] - ba_b["X"], axis=1).max() < 0.5
