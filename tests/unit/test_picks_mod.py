"""detect_2d writes its picks as an IMOD model on the raw stack (etomo's convention for models on a stack)."""

import mrcfile
import numpy as np
import pytest
from imodmodel import ImodModel, read

from jolt import topaz
from jolt.io.imod import write_picks_mod


def test_write_picks_mod_round_trip(tmp_path):
    picks = {0: np.array([[10.0, 20.0], [30.5, 40.0]]), 3: np.array([[5.0, 6.0]])}
    path = tmp_path / "p.mod"
    write_picks_mod(path, picks, (64, 48), 5)
    df = read(path)
    assert len(df) == 3 and sorted(df.contour_id.unique()) == [0, 1]  # one contour per tilt image with picks
    np.testing.assert_allclose(df[["x", "y"]].to_numpy(), [[10.5, 20.5], [31.0, 40.5], [5.5, 6.5]])
    np.testing.assert_array_equal(df.z.to_numpy(), [0.0, 0.0, 3.0])  # integer section index, like etomo's .fid
    model = ImodModel.from_file(path)
    assert (model.header.xmax, model.header.ymax, model.header.zmax) == (64, 48, 5)
    assert model.objects[0].header.flags.scattered
    assert model.objects[0].header.flags.draw_spheres_central_section_only  # only on its own tilt image
    assert model.objects[0].header.pdrawsize == 2
    write_picks_mod(path, picks, (64, 48), 5, sphere_radius=6)
    assert ImodModel.from_file(path).objects[0].header.pdrawsize == 6


def test_write_picks_mod_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        write_picks_mod(tmp_path / "a.mod", {5: np.array([[1.0, 1.0]])}, (8, 8), 5)
    with pytest.raises(ValueError, match="no picks"):
        write_picks_mod(tmp_path / "b.mod", {0: np.zeros((0, 2))}, (8, 8), 5)


def test_detect_tilt_series_writes_star_and_mod(tmp_path, monkeypatch):
    stack = tmp_path / "TS_1.st"
    with mrcfile.new(stack) as mrc:
        mrc.set_data(np.zeros((3, 40, 50), dtype=np.float32))

    def fake_preprocess(install, images, out_dir, device, seed, log_path):
        out_dir.mkdir(parents=True, exist_ok=True)
        return [out_dir / p.name for p in images]

    def fake_extract(install, images, model, radius, threshold, device, seed, out_path, log_path):
        return {"tilt_01": [("12", "7", 1.0), ("3", "4", 0.5)], "tilt_03": [("25", "30", 2.0)]}

    monkeypatch.setattr(topaz, "preprocess", fake_preprocess)
    monkeypatch.setattr(topaz, "extract", fake_extract)
    out_star = tmp_path / "out" / "TS_1_picks.star"
    install = topaz.TopazInstall(executable="topaz", version="test", python=None)
    stats = topaz.detect_tilt_series(
        "TS_1",
        stack,
        3,
        out_star,
        topaz=install,
        model="resnet16",
        radius=4,
        threshold=-6.0,
        top_k=None,
        seed=0,
        device=-1,
        work_dir=tmp_path / "work",
        tilts_dir=None,
        keep_work=False,
        log=lambda m: None,
    )
    df = read(out_star.with_suffix(".mod"))
    assert stats["picks_mod"] == str(out_star.with_suffix(".mod"))
    np.testing.assert_allclose(df[["x", "y", "z"]].to_numpy(), [[12.5, 7.5, 0.0], [3.5, 4.5, 0.0], [25.5, 30.5, 2.0]])
    header = ImodModel.from_file(out_star.with_suffix(".mod")).header
    assert (header.xmax, header.ymax, header.zmax) == (50, 40, 3)
    obj = ImodModel.from_file(out_star.with_suffix(".mod")).objects[0].header
    assert obj.pdrawsize == 4 and obj.flags.draw_spheres_central_section_only
