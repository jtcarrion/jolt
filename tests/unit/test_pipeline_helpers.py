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
