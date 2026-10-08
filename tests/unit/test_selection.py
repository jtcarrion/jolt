import numpy as np

from jolt import selection


def test_residual_selection_threshold_border_and_dedup():
    dims = (100, 200, 200)
    pts = np.array(
        [
            [50.0, 50.0, 50.0],  # kept
            [52.0, 50.0, 50.0],  # duplicate of 0 with a lower residual -> wins
            [2.0, 50.0, 50.0],  # within the 6 px border
            [120.0, 120.0, 50.0],  # residual too high
        ]
    )
    residuals = np.array([3.0, 1.0, 1.0, 20.0])
    kept = selection.residual_selection(
        pts, residuals, 12.0, dedup_radius=5.0, border_padding=6.0, tomogram_dims_zyx=dims
    )
    assert kept.tolist() == [1]


def test_dedup_off_keeps_everything_that_passes():
    pts = np.zeros((3, 3)) + 50.0
    kept = selection.residual_selection(pts, np.ones(3), 12.0, dedup_radius=0.0)
    assert kept.tolist() == [0, 1, 2]


def test_inside_volume_mask_margins():
    dims = (100, 200, 300)
    pts = np.array([[10.0, 10.0, 5.0], [9.9, 50.0, 50.0], [289.0, 189.0, 94.0], [289.1, 50.0, 50.0]])
    assert selection.inside_volume_mask(pts, dims).tolist() == [True, False, True, False]


def test_residual_selection_without_threshold_keeps_every_finite_residual():
    import numpy as np

    from jolt.selection import residual_selection

    pts = np.array([[10.0, 10.0, 10.0], [40.0, 40.0, 20.0], [80.0, 80.0, 30.0]])
    res = np.array([3.0, 50.0, np.inf])
    np.testing.assert_array_equal(residual_selection(pts, res, None, dedup_radius=0.0), [0, 1])
    np.testing.assert_array_equal(residual_selection(pts, res, 12.0, dedup_radius=0.0), [0])
