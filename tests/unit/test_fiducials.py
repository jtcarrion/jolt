"""Gold-bead anchors: reading IMOD .fid models and fixed-correspondence beads in bundle adjustment."""

import numpy as np
import pandas as pd
import torch

from jolt import bundle_adjustment as ba
from jolt.geometry import center_to_corner_2d_bottom_left, corner_to_center_3d, project_points, tilt_projection_matrices
from jolt.io.imod import read_fiducial_beads


def test_fid_beads_map_back_through_the_prealignment(tmp_path):
    import imodmodel

    raw_dims = (200, 180)
    c = np.array([(raw_dims[0] - 1) / 2, (raw_dims[1] - 1) / 2])
    pre = np.array([[1, 0, 0, 1, 3.0, -2.0], [1, 0, 0, 1, -1.5, 4.0], [1, 0, 0, 1, 0.0, 0.0]])
    np.savetxt(tmp_path / "s.prexg", pre)
    raw = {(0, 0): (50.0, 60.0), (0, 1): (51.0, 61.0), (1, 0): (120.0, 30.0), (1, 2): (122.0, 33.0)}
    rows = []
    for (bead, view), (x, y) in raw.items():
        pre_xy = np.array([x, y]) - c + pre[view, 4:6] + c + 0.5  # forward .prexg, then IMOD's +0.5 convention
        rows.append({"object_id": 0, "contour_id": bead, "x": pre_xy[0], "y": pre_xy[1], "z": float(view)})
    imodmodel.write(pd.DataFrame(rows), tmp_path / "s.fid")
    beads = read_fiducial_beads(tmp_path / "s.fid", tmp_path / "s.prexg", 3, raw_dims)
    assert beads["bead_id"].nunique() == 2
    for _, r in beads.iterrows():
        bead = sorted({k[0] for k in raw})[int(r.bead_id)]
        np.testing.assert_allclose([r.x, r.y], raw[(bead, int(r.tilt_index))], atol=1e-3)


def test_bead_anchors_recover_misalignment_better_than_noisy_picks():
    rng = np.random.default_rng(0)
    tomo = (80, 200, 240)
    hw = (200, 240)
    tilts = [float(t) for t in np.arange(-45.0, 45.1, 5.0)]
    proj = tilt_projection_matrices(tilts)

    def project(points_corner):
        center = torch.as_tensor(corner_to_center_3d(points_corner, tomo), dtype=torch.float32)
        return project_points(center, proj).numpy()  # (N, T, 2) internal frame

    points = np.column_stack([rng.uniform(30, 210, 60), rng.uniform(30, 170, 60), rng.uniform(15, 65, 60)])
    beads = np.column_stack([rng.uniform(30, 210, 6), rng.uniform(30, 170, 6), rng.uniform(15, 65, 6)])
    error = rng.uniform(-4, 4, size=(len(tilts), 2))  # alignment error the data carries; BA should find -error...
    p_uv = project(points) + error[None] + rng.normal(0, 1.5, size=(len(points), len(tilts), 2))  # noisy picks
    coords = {}
    for t, tilt in enumerate(tilts):
        corner = center_to_corner_2d_bottom_left(p_uv[:, t], hw)
        coords[tilt] = np.column_stack([corner, np.ones(len(corner))])
    b_uv = project(beads) + error[None]  # beads: exact, same misalignment
    tracks = ba.FiducialTracks(
        tilt_index=np.repeat(np.arange(len(tilts))[None], len(beads), 0).ravel(),
        bead_index=np.repeat(np.arange(len(beads)), len(tilts)),
        uv=b_uv.reshape(-1, 2),
    )
    start = points + rng.normal(0, 1.0, size=points.shape)
    shifts0 = np.zeros((len(tilts), 2), np.float32)

    def gauge_free_error(result):
        d = result["translations"] - error  # recovered shifts should equal the injected error up to a gauge
        d = d - d.mean(axis=0)
        return float(np.sqrt(np.mean(d**2)))

    kw = dict(device=torch.device("cpu"), log=lambda m: None)
    picks_only = ba.greedy_bundle_adjustment(start, coords, tilts, shifts0, hw, tomo, **kw)
    anchored = ba.greedy_bundle_adjustment(start, coords, tilts, shifts0, hw, tomo, fiducials=tracks, **kw)
    assert gauge_free_error(anchored) < 0.5 * gauge_free_error(picks_only)
    assert gauge_free_error(anchored) < 0.3
    assert np.sqrt(np.mean(anchored["fiducial_distances"] ** 2)) < 0.2
