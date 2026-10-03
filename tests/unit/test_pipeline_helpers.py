import pytest

from jolt.pipeline import tilt_index_from_name


@pytest.mark.parametrize(
    ("name", "index"),
    [("TS_01", 1), ("TS_5_31", 31), ("tilt_07.mrc", 7), ("abc_tilt_12_x.tif", 12), ("TS_001_005.mrc", 5)],
)
def test_tilt_index_from_name(name, index):
    assert tilt_index_from_name(name) == index


def test_tilt_index_from_name_rejects_names_without_an_index():
    with pytest.raises(ValueError):
        tilt_index_from_name("TS_5_tilt_-3.00.mrc")


def test_health_ignores_a_common_translation():
    import numpy as np

    from jolt.pipeline import bundle_adjustment_health

    x0 = np.random.default_rng(0).uniform(0, 100, (50, 3))
    h = bundle_adjustment_health(
        x0, x0 + [9.0, -4.0, 2.0], np.zeros((3, 2)), np.zeros((3, 2)), [-3.0, 0.0, 3.0], 8.165, 6.0
    )
    assert h["status"] == "ok"
    assert h["rms_point_displacement_px"] > 8.165 and h["rms_point_displacement_centered_px"] < 1e-9


def test_health_flags_divergence_and_shift_runaway():
    import numpy as np

    from jolt.pipeline import bundle_adjustment_health

    rng = np.random.default_rng(1)
    x0 = rng.uniform(0, 100, (50, 3))
    diverged = bundle_adjustment_health(
        x0, x0 + rng.normal(0, 10, x0.shape), np.zeros((2, 2)), np.zeros((2, 2)), [0.0, 3.0], 8.165, 6.0
    )
    assert diverged["status"] == "diverged"
    shifts = np.array([[0.5, 0.2], [-7.8, -21.3]])
    runaway = bundle_adjustment_health(x0, x0, np.zeros((2, 2)), shifts, [0.0, -3.0], 8.165, 6.0)
    assert runaway["status"] == "shift_runaway" and runaway["runaway_tilts"] == [-3.0]
