"""Synthetic tilt series for tests.

:func:`synthetic_picks` projects random particles exactly (one pick per particle per tilt, Gaussian jitter).
:func:`synthetic_series` mimics real Topaz picks as measured on CZI DS-10441: several species with their own
size, pick jitter and per-tilt detection rate, clustered small particles, satellite picks around large ones,
uniform clutter, non-maximum suppression, and optional injected per-tilt shifts. :func:`exclusive_match_score`
is the CZI benchmark's matching rule (one-to-one within a per-species radius; duplicates are false positives).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from jolt.geometry import center_to_corner_2d_bottom_left, corner_to_center_3d, project_y_axis


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


@dataclass
class Species:
    """One particle type of :func:`synthetic_series` (pixels at the tilt-stack pixel size)."""

    name: str
    diameter_px: float
    score_radius_px: float  # matching radius of :func:`exclusive_match_score`
    abundance: float
    sigma_px: float  # pick jitter
    p_detect: float  # per-tilt detection probability
    satellites: float = 1.0  # extra picks per detected particle per tilt (Poisson mean), beyond the NMS radius
    clustered: bool = False  # placed next to another particle of the same species (12-15 px apart)


#: Three species spanning the CZI DS-10441 range (ferritin 12 px, beta-galactosidase 18 px, ribosome 30 px),
#: with the jitter and detection rates measured against its ground truth (zero-shot picks).
CZI_LIKE_SPECIES = (
    Species("small", 12.0, 6.0, 0.45, 2.7, 0.8, satellites=1.2, clustered=True),
    Species("medium", 18.0, 9.0, 0.25, 2.0, 0.75, satellites=1.3),
    Species("large", 30.0, 15.0, 0.30, 2.4, 0.56, satellites=1.8),
)


@dataclass
class SyntheticSeries:
    """Output of :func:`synthetic_series`; particles and picks in JOLT's frames."""

    xyz_corner: np.ndarray  # (N, 3) particle centers in the internal corner 3D frame (y up)
    species: np.ndarray  # (N,) species names
    picks: dict  # tilt -> (K, 3) picks x, y, score in the STAR corner frame (y down), as JOLT reads them
    shifts: np.ndarray  # (M, 2) injected per-tilt shifts (dx, dy) in the internal frame (y up), zero if none
    tilts: list


def _nms(uv, score, radius, rng):
    order = np.argsort(-score, kind="stable")
    tree = cKDTree(uv)
    alive = np.ones(len(uv), dtype=bool)
    keep = []
    for i in order:
        if alive[i]:
            keep.append(i)
            alive[tree.query_ball_point(uv[i], r=radius)] = False
    return np.asarray(keep, dtype=int)


def synthetic_series(
    n_particles=150,
    image_hw=(384, 384),
    depth=100,
    tilts=None,
    species=CZI_LIKE_SPECIES,
    clutter_per_px2=2.4e-3,
    nms_radius_px=8.0,
    shift_px=0.0,
    border_px=10.0,
    seed=0,
) -> SyntheticSeries:
    """Realistic synthetic picks (see the module docstring). Pick density ends near 3e-3 per px^2 after NMS."""
    rng = np.random.default_rng(seed)
    tilts = list(np.arange(-45.0, 46.0, 4.0)) if tilts is None else list(tilts)
    h, w = image_hw
    tomo_zyx = (depth, h, w)
    abundance = np.array([s.abundance for s in species], dtype=float)
    kinds = rng.choice(len(species), size=n_particles, p=abundance / abundance.sum())
    diam = np.array([species[k].diameter_px for k in kinds])
    lo = np.array([border_px, border_px, border_px])
    hi = np.array([w - 1 - border_px, h - 1 - border_px, depth - 1 - border_px])
    # hard-sphere placement, big particles first, clustered species next to an earlier member of the cluster
    order = np.argsort(-diam, kind="stable")
    placed = np.empty((0, 3))
    placed_diam = np.empty(0)
    placed_cluster = np.empty(0, dtype=bool)
    xyz = np.empty((n_particles, 3))
    for i in order:
        s = species[kinds[i]]
        for _ in range(2000):
            anchors = np.flatnonzero(placed_cluster)
            if s.clustered and anchors.size and rng.random() < 0.9:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                c = np.clip(placed[rng.choice(anchors)] + rng.uniform(12.0, 15.0) * direction, lo, hi)
            else:
                c = rng.uniform(lo, hi)
            if placed.shape[0] == 0 or np.all(np.linalg.norm(placed - c, axis=1) >= 0.5 * (placed_diam + diam[i])):
                break
        else:
            raise RuntimeError("could not place a particle; lower n_particles")
        placed = np.vstack([placed, c])
        placed_diam = np.append(placed_diam, diam[i])
        placed_cluster = np.append(placed_cluster, s.clustered)
        xyz[i] = c
    xyz_c = corner_to_center_3d(xyz.copy(), tomo_zyx)
    u_half, v_half = (w - 1) / 2.0, (h - 1) / 2.0
    shifts = rng.uniform(-shift_px, shift_px, (len(tilts), 2)) if shift_px > 0 else np.zeros((len(tilts), 2))
    picks = {}
    for j, theta in enumerate(tilts):
        proj = project_y_axis(xyz_c, float(theta))
        inside = (np.abs(proj[:, 0]) <= u_half) & (np.abs(proj[:, 1]) <= v_half)
        uv, score = [], []
        for n in np.flatnonzero(inside):
            s = species[kinds[n]]
            if rng.random() < s.p_detect:
                uv.append(proj[n] + rng.normal(0.0, s.sigma_px, 2))
                score.append(rng.normal(1.0, 1.0))
                for _ in range(rng.poisson(s.satellites)):
                    r = rng.uniform(nms_radius_px + 0.5, max(nms_radius_px + 2.0, 0.75 * s.diameter_px))
                    a = rng.uniform(0.0, 2.0 * np.pi)
                    uv.append(proj[n] + r * np.array([np.cos(a), np.sin(a)]))
                    score.append(rng.normal(0.0, 1.0))
        n_clutter = rng.poisson(clutter_per_px2 * h * w)
        clutter = np.column_stack([rng.uniform(-u_half, u_half, n_clutter), rng.uniform(-v_half, v_half, n_clutter)])
        uv = np.vstack([np.asarray(uv).reshape(-1, 2), clutter]) + shifts[j]
        score = np.concatenate([np.asarray(score, dtype=float), rng.normal(0.0, 1.0, n_clutter)])
        keep = (np.abs(uv[:, 0]) <= u_half) & (np.abs(uv[:, 1]) <= v_half)
        uv, score = uv[keep], score[keep]
        if nms_radius_px > 0 and len(uv) > 1:
            k = _nms(uv, score, nms_radius_px, rng)
            uv, score = uv[k], score[k]
        corner = center_to_corner_2d_bottom_left(uv, image_hw)
        picks[float(theta)] = np.column_stack([corner, score])
    return SyntheticSeries(xyz, np.array([species[k].name for k in kinds]), picks, shifts, tilts)


def exclusive_match_score(candidates, truth_xyz, truth_species, radius_by_species):
    """One-to-one matching within each true particle's species radius; every unmatched candidate is a false
    positive (duplicates included). Returns ``tp, fp, fn, recall, precision``."""
    cand = np.asarray(candidates, dtype=float).reshape(-1, 3)
    truth = np.asarray(truth_xyz, dtype=float)
    radius = np.array([radius_by_species[s] for s in truth_species], dtype=float)
    if len(cand) == 0 or len(truth) == 0:
        return 0, len(cand), len(truth), 0.0, 0.0
    tree = cKDTree(truth)
    lists = tree.query_ball_point(cand, r=float(radius.max()))
    ci = [i for i, c in enumerate(lists) if c]
    tj = sorted({j for c in lists for j in c})
    tp = 0
    if ci:
        pos = {j: k for k, j in enumerate(tj)}
        cost = np.full((len(ci), len(tj)), 1e9)
        for r, i in enumerate(ci):
            for j in lists[i]:
                d = np.linalg.norm(cand[i] - truth[j])
                if d <= radius[j]:
                    cost[r, pos[j]] = d
        rows, cols = linear_sum_assignment(cost)
        tp = int((cost[rows, cols] < 1e9).sum())
    fp, fn = len(cand) - tp, len(truth) - tp
    return tp, fp, fn, tp / len(truth), tp / len(cand)
