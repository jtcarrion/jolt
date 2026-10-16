"""Per-series pick noise model: the EM fit recovers known parameters; the estimator bounds and falls back."""

import numpy as np

from jolt import mlesac
from jolt.geometry import center_to_corner_2d_bottom_left, project_y_axis
from tests.synthetic import synthetic_picks

IMAGE_HW = (256, 256)


def _mixture_sample(rng, n, sigma, gamma, rho):
    """Nearest-pick distances from the model: Rayleigh(sigma) with probability gamma, else Poisson NN(rho)."""
    inlier = rng.random(n) < gamma
    d_in = sigma * np.sqrt(-2.0 * np.log(rng.random(n)))
    d_out = np.sqrt(-np.log(rng.random(n)) / (np.pi * rho))
    return np.where(inlier, d_in, d_out)


def test_em_recovers_sigma_and_gamma():
    rng = np.random.default_rng(0)
    sigma, gamma, rho = 2.5, 0.7, 3.0e-3
    d = _mixture_sample(rng, 40_000, sigma, gamma, rho)
    s_hat, g_hat, iterations, converged = mlesac.fit_rayleigh_poisson_mixture(d, np.full(d.size, rho))
    assert converged and iterations < 200
    assert abs(s_hat - sigma) / sigma < 0.05
    assert abs(g_hat - gamma) < 0.05


def test_em_with_per_tilt_density():
    rng = np.random.default_rng(1)
    rhos = np.repeat([1.5e-3, 3.0e-3, 4.5e-3], 10_000)
    d = np.concatenate([_mixture_sample(rng, 10_000, 3.2, 0.5, r) for r in (1.5e-3, 3.0e-3, 4.5e-3)])
    s_hat, g_hat, _, converged = mlesac.fit_rayleigh_poisson_mixture(d, rhos)
    assert converged
    assert abs(s_hat - 3.2) / 3.2 < 0.05
    assert abs(g_hat - 0.5) < 0.05


def _picks_with_clutter(rng, n_clutter, sigma_true, p_detect):
    """Synthetic series: true picks jittered by sigma_true, kept with probability p_detect, plus uniform clutter."""
    pts_center, picks, tilts = synthetic_picks(n_particles=60, image_hw=IMAGE_HW, depth=60, noise_px=0.0, seed=2)
    h, w = IMAGE_HW
    out = {}
    for t in tilts:
        uv = project_y_axis(pts_center, float(t)) + rng.normal(0.0, sigma_true, (len(pts_center), 2))
        keep = rng.random(len(uv)) < p_detect
        clutter = np.column_stack([rng.uniform(-w / 2, w / 2, n_clutter), rng.uniform(-h / 2, h / 2, n_clutter)])
        corner = center_to_corner_2d_bottom_left(np.vstack([uv[keep], clutter]), IMAGE_HW)
        out[float(t)] = np.column_stack([corner, np.zeros(len(corner))])
    return pts_center, out


def test_estimate_noise_model_on_synthetic_series():
    rng = np.random.default_rng(3)
    pts_center, coords = _picks_with_clutter(rng, n_clutter=150, sigma_true=2.5, p_detect=0.7)
    est = mlesac.estimate_noise_model(pts_center, coords, IMAGE_HW, sigma_fallback=4.0, gamma_fallback=0.7)
    assert not est.fallback and est.converged
    assert est.n_points == 60 and est.n_distances >= 500
    assert abs(est.sigma_px - 2.5) / 2.5 < 0.15
    assert abs(est.gamma_raw - 0.7) < 0.1
    assert est.gamma == est.gamma_raw  # inside the gamma bounds: used as is


def test_estimate_noise_model_clamps_and_falls_back():
    rng = np.random.default_rng(4)
    # clean picks (0.3 px jitter): sigma below the lower bound -> fallback, reason says so
    pts_center, coords = _picks_with_clutter(rng, n_clutter=20, sigma_true=0.3, p_detect=1.0)
    est = mlesac.estimate_noise_model(pts_center, coords, IMAGE_HW, sigma_fallback=2.5, gamma_fallback=0.7)
    assert est.fallback and est.sigma_px == 2.5 and est.gamma == 0.7 and "outside" in est.reason
    assert est.sigma_raw_px < 1.5
    # too few hypotheses -> fallback before any fit
    est = mlesac.estimate_noise_model(pts_center[:5], coords, IMAGE_HW, sigma_fallback=2.5, gamma_fallback=0.7)
    assert est.fallback and est.n_distances == 0 and "hypotheses" in est.reason
