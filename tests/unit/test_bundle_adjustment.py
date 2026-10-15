"""Bundle adjustment: the evidence loss (gradient, limits, convergence) and the nearest loss still runs (CPU)."""

import numpy as np
import pytest
import torch

from jolt import bundle_adjustment as ba
from jolt.geometry import (
    center_to_corner_2d_bottom_left,
    corner_to_center_2d_bottom_left,
    corner_to_center_3d,
    project_y_axis,
    tilt_projection_matrices,
)
from tests.synthetic import synthetic_picks

IMAGE_HW = (256, 256)
DEPTH = 60
TOMO_ZYX = (DEPTH, IMAGE_HW[0], IMAGE_HW[1])


def _with_clutter(picks, n_clutter, rng):
    """Add uniformly distributed false picks to every tilt (STAR corner frame)."""
    h, w = IMAGE_HW
    out = {}
    for t, p in picks.items():
        clutter = np.column_stack(
            [rng.uniform(0, w - 1, n_clutter), rng.uniform(0, h - 1, n_clutter), rng.normal(size=n_clutter)]
        )
        out[t] = np.vstack([p, clutter])
    return out


def _run(loss, points_corner, picks, tilts, shifts0=None, **overrides):
    settings = ba.BundleAdjustmentSettings(loss=loss, **overrides)
    shifts0 = np.zeros((len(tilts), 2), dtype=np.float32) if shifts0 is None else shifts0
    return ba.greedy_bundle_adjustment(
        points_corner,
        picks,
        tilts,
        shifts0,
        IMAGE_HW,
        TOMO_ZYX,
        settings,
        device=torch.device("cpu"),
        log=lambda *a: None,
    )


def test_soft_inside_values():
    inside = ba.soft_inside(torch.tensor([[0.0, 0.0]]), IMAGE_HW, 3.0)
    on_border = ba.soft_inside(torch.tensor([[127.5, 0.0]]), IMAGE_HW, 3.0)
    far_outside = ba.soft_inside(torch.tensor([[200.0, 0.0]]), IMAGE_HW, 3.0)
    assert float(inside) == pytest.approx(1.0, abs=1e-6)
    assert float(on_border) == pytest.approx(0.5, abs=1e-6)
    assert float(far_outside) < 1e-6


def test_pick_neighbourhoods_lists_every_pick_within_radius():
    projected = np.zeros((2, 2, 2))
    projected[1, :, :] = 50.0
    picks = [np.array([[1.0, 0.0], [49.0, 50.0], [30.0, 30.0]]), None]
    n, t, k = ba.pick_neighbourhoods(projected, picks, radius=5.0)
    assert set(zip(n.tolist(), t.tolist(), k.tolist(), strict=True)) == {(0, 0, 0), (1, 0, 1)}


def test_evidence_terms_no_neighbour_is_the_miss_penalty():
    projected = torch.zeros((3, 2, 2))
    empty = torch.empty(0, dtype=torch.long)
    evidence, share = ba.evidence_terms(
        projected, torch.zeros((0, 2)), empty, empty, empty, sigma_px=2.5, gamma=0.7, radius_px=12.0
    )
    assert torch.allclose(evidence, torch.full((3, 2), 0.3))
    assert torch.all(share == 0)


def test_evidence_gradient_matches_finite_differences():
    """Autograd of the evidence data term vs central differences, float64, picks within and beyond the radius."""
    rng = np.random.default_rng(0)
    tilts = [-45.0, -15.0, 15.0, 45.0]
    n = 5
    pts = np.column_stack([rng.uniform(60, 190, n), rng.uniform(60, 190, n), rng.uniform(10, 50, n)])
    picks_np = []
    for t in tilts:
        uv = project_y_axis(corner_to_center_3d(pts.copy(), TOMO_ZYX), t) + rng.normal(0, 2.0, (n, 2))
        picks_np.append(np.vstack([uv, rng.uniform(-100, 100, (12, 2))]))
    picks_flat = torch.tensor(np.vstack(picks_np), dtype=torch.float64)
    offsets = np.cumsum([0] + [len(p) for p in picks_np])
    proj_m = tilt_projection_matrices(tilts, dtype=torch.float64)
    x = torch.tensor(corner_to_center_3d(pts.copy(), TOMO_ZYX), dtype=torch.float64, requires_grad=True)
    d = torch.zeros(len(tilts), 2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        projected0 = (
            torch.einsum("mij,nj->nmi", proj_m, torch.cat([x, torch.ones(n, 1, dtype=torch.float64)], 1))[:, :, :2] + d
        ).numpy()
    n_idx, t_idx, k_idx = ba.pick_neighbourhoods(projected0, picks_np, radius=12.0)
    n_idx, t_idx = torch.as_tensor(n_idx), torch.as_tensor(t_idx)
    k_flat = torch.as_tensor(offsets[t_idx.numpy()] + k_idx)

    def loss_fn(x, d):
        homogeneous = torch.cat([x, torch.ones(n, 1, dtype=torch.float64)], dim=1)
        projected = torch.einsum("mij,nj->nmi", proj_m, homogeneous)[:, :, :2] + d.unsqueeze(0)
        evidence, _ = ba.evidence_terms(
            projected, picks_flat, n_idx, t_idx, k_flat, sigma_px=2.5, gamma=0.7, radius_px=12.0
        )
        return -(ba.soft_inside(projected, IMAGE_HW, 3.0) * torch.log(evidence)).sum()

    loss = loss_fn(x, d)
    gx, gd = torch.autograd.grad(loss, [x, d])
    eps = 1e-5
    for param, grad in ((x, gx), (d, gd)):
        flat = param.detach().reshape(-1)
        for i in range(flat.numel()):
            plus, minus = flat.clone(), flat.clone()
            plus[i] += eps
            minus[i] -= eps
            if param is x:
                fd = (loss_fn(plus.view_as(x), d) - loss_fn(minus.view_as(x), d)) / (2 * eps)
            else:
                fd = (loss_fn(x, plus.view_as(d)) - loss_fn(x, minus.view_as(d))) / (2 * eps)
            assert float(grad.reshape(-1)[i]) == pytest.approx(float(fd.detach()), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("loss", ["evidence", "nearest"])
def test_both_losses_recover_particles_and_shifts(loss):
    """From particles perturbed by 2 px and picks shifted per tilt, both losses recover positions and shifts."""
    rng = np.random.default_rng(1)
    pts_center, picks, tilts = synthetic_picks(n_particles=25, image_hw=IMAGE_HW, depth=DEPTH, noise_px=0.3, seed=1)
    shifts_true = rng.uniform(-3.0, 3.0, (len(tilts), 2))
    shifted = {}
    for i, t in enumerate(tilts):
        uv = corner_to_center_2d_bottom_left(picks[float(t)][:, :2], IMAGE_HW) + shifts_true[i]
        shifted[float(t)] = np.column_stack([center_to_corner_2d_bottom_left(uv, IMAGE_HW), picks[float(t)][:, 2]])
    shifted = _with_clutter(shifted, 40, rng)
    pts_corner = pts_center.copy()
    pts_corner[:, 0] += (IMAGE_HW[1] - 1) / 2.0
    pts_corner[:, 1] += (IMAGE_HW[0] - 1) / 2.0
    pts_corner[:, 2] += (DEPTH - 1) / 2.0
    start = pts_corner + rng.normal(0, 2.0, pts_corner.shape)
    result = _run(loss, start, shifted, [float(t) for t in tilts])
    assert result["loss"] == loss
    err = np.linalg.norm(result["X"] - pts_corner, axis=1)
    assert np.median(err) < 0.75, f"{loss}: median position error {np.median(err):.2f} px"
    shift_err = result["translations"] - shifts_true
    shift_err -= shift_err.mean(axis=0)  # the common shift is a gauge
    assert np.sqrt(np.mean(shift_err**2)) < 0.5, f"{loss}: shift error {np.sqrt(np.mean(shift_err**2)):.2f} px"


def test_evidence_result_keys_and_support():
    pts_center, picks, tilts = synthetic_picks(n_particles=15, image_hw=IMAGE_HW, depth=DEPTH, seed=2)
    pts_corner = pts_center + np.array([(IMAGE_HW[1] - 1) / 2.0, (IMAGE_HW[0] - 1) / 2.0, (DEPTH - 1) / 2.0])
    result = _run("evidence", pts_corner, picks, [float(t) for t in tilts])
    for key in ("X", "translations", "rmse", "steps", "converged", "match_statistics", "soft_support", "log_evidence"):
        assert key in result
    assert result["soft_support"].shape == (15,)
    # every synthetic particle is picked in every tilt: the soft support is close to the number of tilts
    assert np.all(result["soft_support"] > 0.9 * len(tilts))
    assert result["sigma_px"] == 2.5 and result["gamma"] == 0.7


def test_evidence_is_deterministic_under_tiny_perturbation():
    rng = np.random.default_rng(3)
    pts_center, picks, tilts = synthetic_picks(n_particles=20, image_hw=IMAGE_HW, depth=DEPTH, seed=3)
    picks = _with_clutter(picks, 60, rng)
    pts_corner = pts_center + np.array([(IMAGE_HW[1] - 1) / 2.0, (IMAGE_HW[0] - 1) / 2.0, (DEPTH - 1) / 2.0])
    start = pts_corner + rng.normal(0, 1.5, pts_corner.shape)
    a = _run("evidence", start, picks, [float(t) for t in tilts])
    b = _run("evidence", start + rng.normal(0, 0.001, start.shape), picks, [float(t) for t in tilts])
    assert np.abs(a["translations"] - b["translations"]).max() < 0.02
    assert np.linalg.norm(a["X"] - b["X"], axis=1).max() < 0.05


def test_unknown_loss_raises():
    pts_center, picks, tilts = synthetic_picks(n_particles=5, image_hw=IMAGE_HW, depth=DEPTH)
    with pytest.raises(ValueError, match="unknown bundle-adjustment loss"):
        _run("sinkhorn", np.zeros((5, 3)), picks, [float(t) for t in tilts])
