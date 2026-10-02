import numpy as np
import pytest
import torch

from jolt import geometry


def test_center_corner_round_trip():
    dims = (200, 630, 630)
    pts = np.array([[0.0, 0.0, 0.0], [629.0, 629.0, 199.0], [314.5, 314.5, 99.5]])
    center = geometry.corner_to_center_3d(pts, dims)
    np.testing.assert_allclose(center[2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(geometry.center_to_corner_3d(center, dims), pts)


def test_bottom_left_frame_round_trip():
    hw = (560, 788)
    picks = np.array([[0.0, 0.0], [787.0, 559.0], [10.5, 20.25]])
    center = geometry.corner_to_center_2d_bottom_left(picks, hw)
    assert center[0, 1] == pytest.approx((559.0 - 0.0) - 279.5)  # y flips: top row is +v
    np.testing.assert_allclose(geometry.center_to_corner_2d_bottom_left(center, hw), picks)


def test_raw_picks_identity_xf_is_noop_for_equal_canvases():
    picks = np.array([[100.0, 200.0, 1.5], [3.0, 4.0, -2.0]])
    row = {"A11": 1.0, "A12": 0.0, "A21": 0.0, "A22": 1.0, "DX": 0.0, "DY": 0.0}
    np.testing.assert_allclose(geometry.raw_picks_to_aligned(picks, row, (630, 630), (630, 630)), picks)


def test_raw_picks_rotation_and_canvas_change():
    # 90 degree rotation of a landscape raw image onto a portrait aligned canvas.
    row = {"A11": 0.0, "A12": -1.0, "A21": 1.0, "A22": 0.0, "DX": 2.0, "DY": -1.0}
    out = geometry.raw_picks_to_aligned(np.array([[394.0 + 10.0, 280.0, 0.0]]), row, (788, 560), (560, 788))
    # x' = A (x - c_raw) + d + c_out with c_raw = (394, 280), c_out = (280, 394)
    np.testing.assert_allclose(out[0, :2], [0.0 * 10 - 1.0 * 0 + 2.0 + 280.0, 1.0 * 10 + 0.0 - 1.0 + 394.0])


def test_projection_matrices_follow_the_tilt_model():
    tilts = [-60.0, 0.0, 33.0]
    mats = geometry.tilt_projection_matrices(tilts)
    pts = torch.tensor([[10.0, -5.0, 20.0]])
    uv = geometry.project_points(pts, mats)[0].numpy()
    for k, theta in enumerate(np.deg2rad(tilts)):
        assert uv[k, 0] == pytest.approx(np.cos(theta) * 10.0 + np.sin(theta) * 20.0, abs=1e-4)
        assert uv[k, 1] == pytest.approx(-5.0, abs=1e-5)
    np.testing.assert_allclose(geometry.project_y_axis(np.array([10.0, -5.0, 20.0]), 33.0), uv[2], atol=1e-4)


def test_aligned_to_raw_inverts_raw_to_aligned():
    row = {"A11": 0.98, "A12": -0.17, "A21": 0.17, "A22": 0.98, "DX": 3.5, "DY": -7.25}
    raw = np.array([[100.0, 200.0, 0.0], [700.0, 50.0, 0.0]])
    aligned = geometry.raw_picks_to_aligned(raw, row, (788, 560), (560, 788))
    np.testing.assert_allclose(geometry.aligned_to_raw(aligned[:, :2], row, (788, 560), (560, 788)), raw[:, :2])


def test_star_points_project_like_the_pipeline_frames():
    # A particle in the STAR frame (y down) projects at 0 deg onto its own (x, y) in the aligned image.
    row = {"A11": 1.0, "A12": 0.0, "A21": 0.0, "A22": 1.0, "DX": 0.0, "DY": 0.0}
    xy = geometry.project_star_points_to_raw(np.array([[120.0, 40.0, 10.0]]), 0.0, row, (256, 256, 60), (256, 256))
    np.testing.assert_allclose(xy, [[120.0, 40.0]], atol=1e-9)
    # At 30 deg x moves by cos/sin of the centered x and z; y is unchanged.
    xy = geometry.project_star_points_to_raw(np.array([[120.0, 40.0, 10.0]]), 30.0, row, (256, 256, 60), (256, 256))
    t = np.deg2rad(30.0)
    np.testing.assert_allclose(xy[0, 0], np.cos(t) * (120 - 127.5) + np.sin(t) * (10 - 29.5) + 127.5, atol=1e-9)
    np.testing.assert_allclose(xy[0, 1], 40.0, atol=1e-9)
