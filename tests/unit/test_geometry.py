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
    c_raw, c_out = np.array([393.5, 279.5]), np.array([279.5, 393.5])  # (N - 1) / 2
    out = geometry.raw_picks_to_aligned(np.array([[c_raw[0] + 10.0, c_raw[1], 0.0]]), row, (788, 560), (560, 788))
    np.testing.assert_allclose(out[0, :2], [0.0 + 2.0 + c_out[0], 10.0 - 1.0 + c_out[1]])


# Translations measured from IMOD 5.1.9 ``newstack -xf -linear`` on Gaussian blobs at known positions
# (raw size, aligned size, xf, measured translation t = aligned - A @ raw). IMOD rotates about (N - 1) / 2 in
# 0-based pixel coordinates; W / 2 would be off by up to ~0.45 px.
ROT30 = [np.cos(0.5236), -np.sin(0.5236), np.sin(0.5236), np.cos(0.5236)]
IMOD_MEASURED = [
    ((63, 63), (63, 63), [*ROT30, 1.5, 2.5], (21.151, -8.845)),
    ((64, 64), (64, 64), [*ROT30, 1.5, 2.5], (21.470, -9.030)),
    ((80, 60), (70, 90), [*ROT30, -2.0, 1.0], (13.040, 0.204)),
    ((63, 47), (47, 63), [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], (46.0, 0.0)),
]


@pytest.mark.parametrize(("raw", "aligned", "xf", "measured"), IMOD_MEASURED)
def test_raw_to_aligned_matches_imod_newstack(raw, aligned, xf, measured):
    row = dict(zip(("A11", "A12", "A21", "A22", "DX", "DY"), xf, strict=True))
    point = np.array([[20.0, 18.0, 0.0]])
    out = geometry.raw_picks_to_aligned(point, row, raw, aligned)[0, :2]
    a = np.array([[xf[0], xf[1]], [xf[2], xf[3]]])
    np.testing.assert_allclose(out - a @ point[0, :2], measured, atol=0.01)


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


def test_teamtomo_frame_shifts_z_by_half_a_voxel_only_for_even_depth():
    pts = np.array([[10.0, 20.0, 30.0], [300.0, 5.5, 99.25]])
    even = geometry.output_frame_transform("teamtomo", (652, 630, 340), 10.0)
    odd = geometry.output_frame_transform("teamtomo", (652, 630, 341), 10.0)
    np.testing.assert_allclose(even.to_frame(pts), pts + [0.0, 0.0, 0.5])
    np.testing.assert_allclose(odd.to_frame(pts), pts)


def test_warp_frame_without_xml_dimensions_is_a_flip_about_nz_over_two():
    f = geometry.output_frame_transform("warp", (652, 630, 340), 10.0)
    pts = np.array([[10.0, 20.0, 30.0], [651.0, 629.0, 0.0]])
    expected = pts.copy()
    expected[:, 2] = 340 - 0.5 - pts[:, 2]
    np.testing.assert_allclose(f.to_frame(pts), expected)
    assert "exact binning" in f.source


def test_warp_frame_matches_the_empiar_10499_bead_measurement():
    # TS_00254: ImageDimensionsAngstrom 6526.519 x 6308.855, DimensionsAngstrom ... x 3401, 10 A/px, 652 x 630 x 340.
    # Measured over 50 gold beads (Warp tomogram minus JOLT/IMOD frame, z mapped back as NZ - 1 - z):
    # x +0.30 +0.00088 (x - 326), y +0.43 +0.00114 (y - 315), z -0.52 +0.0012 (z - 169.5).
    f = geometry.output_frame_transform(
        "warp",
        (652, 630, 340),
        10.0,
        warp_image_dims_angstrom=(6526.519, 6308.855),
        warp_volume_dims_angstrom=(6526.519, 6308.855, 3401.0),
        image_rotation_deg=4.3,
    )
    pts = np.array([[326.0, 315.0, 169.5], [28.0, 45.0, 65.0], [626.0, 604.0, 300.0]])
    out = f.to_frame(pts)
    measured = np.column_stack(
        [
            pts[:, 0] + 0.30 + 0.00088 * (pts[:, 0] - 326),
            pts[:, 1] + 0.43 + 0.00114 * (pts[:, 1] - 315),
            339 - (pts[:, 2] - 0.52 + 0.0012 * (pts[:, 2] - 169.5)),
        ]
    )
    np.testing.assert_allclose(out, measured, atol=0.15)


def test_output_frames_round_trip():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 600, size=(50, 3))
    for frame in geometry.OUTPUT_FRAMES:
        f = geometry.output_frame_transform(
            frame,
            (652, 630, 340),
            10.0,
            warp_image_dims_angstrom=(6526.5, 6308.9),
            warp_volume_dims_angstrom=(6526.5, 6308.9, 3401.0),
            image_rotation_deg=87.0,
        )
        np.testing.assert_allclose(f.from_frame(f.to_frame(pts)), pts, atol=1e-9)
