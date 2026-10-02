from pathlib import Path

import pytest

from jolt.io import config, imod, warp
from jolt.io.star import read_star_table

DATA = Path(__file__).parents[1] / "data" / "czi_TS_5"
STACK = DATA / "warp_tiltseries" / "tiltstack" / "TS_5"


def test_read_xf_attaches_tilts_in_file_order():
    angles = imod.read_tlt(STACK / "TS_5.tlt")
    rows = imod.read_xf(STACK / "TS_5.xf", angles)
    assert [r["TILT"] for r in rows] == angles
    assert rows[0]["A11"] == pytest.approx(1.0)


def test_read_xf_rejects_count_mismatch():
    angles = imod.read_tlt(STACK / "TS_5.tlt")
    with pytest.raises(ValueError, match="positional"):
        imod.read_xf(STACK / "TS_5.xf", angles[:-1])


def test_descending_order_is_kept(tmp_path):
    (tmp_path / "d.tlt").write_text("3.0\n0.0\n-3.0\n\n")
    (tmp_path / "d.xf").write_text("1 0 0 1 1.5 0\n1 0 0 1 0 0\n1 0 0 1 -2 1\n")
    rows = imod.read_xf(tmp_path / "d.xf", imod.read_tlt(tmp_path / "d.tlt"))
    assert [r["TILT"] for r in rows] == [3.0, 0.0, -3.0]
    assert [r["DX"] for r in rows] == [1.5, 0.0, -2.0]


def test_blank_line_inside_file_is_an_error(tmp_path):
    (tmp_path / "b.tlt").write_text("3.0\n\n-3.0\n")
    with pytest.raises(ValueError, match="blank line"):
        imod.read_tlt(tmp_path / "b.tlt")


def test_write_xf_round_trip(tmp_path):
    angles = imod.read_tlt(STACK / "TS_5.tlt")
    rows = imod.read_xf(STACK / "TS_5.xf", angles)
    rows[3]["DX"] = 1.23456789
    imod.write_xf(tmp_path / "out.xf", rows)
    back = imod.read_xf(tmp_path / "out.xf", angles)
    assert back[3]["DX"] == pytest.approx(1.2345679, abs=1e-7)
    assert len(back) == len(rows)


def test_warp_discovery_on_test_data():
    (series,) = warp.discover(DATA)
    assert series.name == "TS_5"
    assert series.n_tilts == 31
    assert series.raw_dims_xy == (630, 630)
    assert series.pixel_size == pytest.approx(10.0)
    assert series.warnings == []


def test_warp_discovery_unknown_name():
    with pytest.raises(FileNotFoundError):
        warp.discover(DATA, names=["TS_99"])


def test_picks_star_names_are_strings():
    table = read_star_table(STACK / "TS_5_topaz_picks.star", required=["rlnMicrographName", "rlnCoordinateX"])
    assert table["rlnMicrographName"].iloc[0] == "TS_01"


def test_config_round_trip(tmp_path):
    cfg = {
        "config_version": 1,
        "project": {"warp_dir": str(DATA), "tomo_dims_xyz": [630, 630, 200]},
        "tilt_series": {
            "TS_5": {
                "raw_stack": "a.st",
                "xf": "a.xf",
                "tlt": "a.tlt",
                "n_tilts": 31,
                "raw_dims_xy": [630, 630],
                "pixel_size": 10.0,
            }
        },
    }
    config.record_command(cfg, "import", {"warp_dir": DATA, "overwrite": False})
    config.save(tmp_path, cfg)
    loaded = config.load(tmp_path)
    assert loaded["commands"]["import"]["parameters"]["warp_dir"] == str(DATA)
    text = (tmp_path / "jolt.yaml").read_text()
    assert "git" not in text and "version" not in text.replace("config_version", "")


def test_config_rejects_bad_dims(tmp_path):
    with pytest.raises(ValueError, match="tomo_dims_xyz"):
        config.validate({"config_version": 1, "project": {"tomo_dims_xyz": [1, 2]}, "tilt_series": {}})
