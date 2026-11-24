"""The warp output frame takes the tomogram size from the WarpTools settings, not from the XML cache."""

from pathlib import Path

import numpy as np
import pytest

from jolt import pipeline
from jolt.io import warp
from tests.synthetic import write_warp_settings

# bmp6: 4092 x 5760 x 2800 px at 1.368 A (Warp settings), tilt stacks at 10 A/px, tomograms 560 x 788 x 384.
BMP6_SETTINGS = (1.368, (4092, 5760, 2800))
BMP6_TOMO = (560, 788, 384)
BMP6_IMAGE_A = (7879.68, 5597.856)  # the raw tilt images (5760 x 4092 px), stored by Warp in the XML


def _xml(path, n=3, image=None, volume=None):
    import warpxmlfile

    attributes = {}
    if image is not None:
        attributes["ImageDimensionsAngstrom"] = ", ".join(f"{v}" for v in image)
    if volume is not None:
        attributes["VolumeDimensionsAngstrom"] = ", ".join(f"{v}" for v in volume)
    xml = warpxmlfile.WarpTiltSeries(
        attributes=attributes,
        angles=np.linspace(-30, 30, n),
        use_tilt=np.ones(n, dtype=bool),
        axis_angle=np.zeros(n),
        axis_offset_x=np.zeros(n),
        axis_offset_y=np.zeros(n),
    )
    warpxmlfile.write(xml, path)
    return path


def _alignment(folder, n=3):
    (folder / "s.tlt").write_text("".join(f"{t:.2f}\n" for t in np.linspace(-30, 30, n)))
    (folder / "s.xf").write_text("1.0 0.0 0.0 1.0 0.0 0.0\n" * n)
    return folder / "s.xf", folder / "s.tlt"


def test_read_settings_with_byte_order_mark(tmp_path):
    write_warp_settings(tmp_path / "w.settings", *BMP6_SETTINGS)
    s = warp.read_settings(tmp_path / "w.settings")
    assert s.pixel_size == 1.368 and s.tomo_dims_xyz == (4092, 5760, 2800)
    np.testing.assert_allclose(s.volume_angstrom, (5597.856, 7879.68, 3830.4))


@pytest.mark.parametrize(
    "text, message",
    [
        ("<Settings><Import><Param Name='PixelSize' Value='1' /></Import></Settings>", "no Tomo.DimensionsX"),
        (
            "<Settings><Import><Param Name='PixelSize' Value='0' /></Import><Tomo>"
            "<Param Name='DimensionsX' Value='1' /><Param Name='DimensionsY' Value='1' />"
            "<Param Name='DimensionsZ' Value='1' /></Tomo></Settings>",
            "must be positive",
        ),
        ("not xml", "cannot read it"),
    ],
)
def test_read_settings_rejects_incomplete_files(tmp_path, text, message):
    (tmp_path / "bad.settings").write_text(text)
    with pytest.raises(ValueError, match=message):
        warp.read_settings(tmp_path / "bad.settings")


def test_find_settings(tmp_path):
    assert warp.find_settings(tmp_path)[0] is None
    write_warp_settings(tmp_path / "a_variant.settings", *BMP6_SETTINGS)
    s, why = warp.find_settings(tmp_path)
    assert s.path.name == "a_variant.settings" and "only tilt-series settings file" in why
    # several files that agree: warp_tiltseries.settings is preferred
    write_warp_settings(tmp_path / "warp_tiltseries.settings", *BMP6_SETTINGS)
    s, why = warp.find_settings(tmp_path)
    assert s.path.name == "warp_tiltseries.settings" and "2 tilt-series settings files" in why
    # one that disagrees: no choice is made
    write_warp_settings(tmp_path / "z_other.settings", 1.368, (4092, 5760, 2000))
    s, why = warp.find_settings(tmp_path)
    assert s is None and "disagree" in why and "z_other.settings" in why


def test_physical_dimensions_treats_all_zero_as_missing(tmp_path):
    xml = _xml(tmp_path / "TS.xml", image=(0, 0), volume=(0, 0, 0))
    assert warp.physical_dimensions(xml) == (None, None)
    xml = _xml(tmp_path / "TS2.xml", image=BMP6_IMAGE_A, volume=(5597.856, 7879.68, 1))
    assert warp.physical_dimensions(xml) == (BMP6_IMAGE_A, (5597.856, 7879.68, 1.0))


def test_warp_frame_uses_settings_not_the_xml_placeholder(tmp_path):
    """ts_import_alignments leaves DimensionsAngstrom Z at 1 A; ts_ctf writes the real value. Same frame either way."""
    write_warp_settings(tmp_path / "w.settings", *BMP6_SETTINGS)
    settings = warp.read_settings(tmp_path / "w.settings")
    xf, tlt = _alignment(tmp_path)
    placeholder = _xml(tmp_path / "after_import.xml", image=BMP6_IMAGE_A, volume=(5597.856, 7879.68, 1))
    real = _xml(tmp_path / "after_ctf.xml", image=BMP6_IMAGE_A, volume=(5597.856, 7879.68, 3830.4001))
    frames = [
        pipeline.output_frame_for("warp", BMP6_TOMO, 10.0, xf, tlt, xml, warp_settings=settings)
        for xml in (placeholder, real)
    ]
    pts = np.array([[280.0, 394.0, 191.5], [10.0, 20.0, 5.0], [550.0, 780.0, 380.0]])
    np.testing.assert_allclose(frames[0].to_frame(pts), frames[1].to_frame(pts))
    for f in frames:
        assert f.volume_source == "settings" and "w.settings" in f.source
        # Warp's center is 3830.4 A / 2 / 10 A = 191.52 voxels (not 192.0 from --tomo_dims_xyz Z = 384)
        np.testing.assert_allclose(f.center_out, (279.8928, 393.984, 191.52), atol=1e-4)
    assert frames[0].describe()["volume_source"] == "settings"


def test_warp_frame_requires_settings(tmp_path):
    xf, tlt = _alignment(tmp_path)
    with pytest.raises(ValueError, match="needs the WarpTools settings"):
        pipeline.output_frame_for("warp", BMP6_TOMO, 10.0, xf, tlt, None)
    # teamtomo does not need them
    assert pipeline.output_frame_for("teamtomo", BMP6_TOMO, 10.0, xf, tlt, None).volume_source == "none"


def test_resolve_project_settings(tmp_path):
    write_warp_settings(tmp_path / "warp_tiltseries.settings", *BMP6_SETTINGS)
    project = {"warp_dir": str(tmp_path)}
    s, how, new = warp.resolve_project_settings(project)  # an older jolt.yaml: found in the Warp folder
    assert s.tomo_dims_xyz == (4092, 5760, 2800) and new and "not recorded" in how
    project["warp_settings"] = warp.settings_record(s)
    s2, how, new = warp.resolve_project_settings(project)  # recorded by jolt import
    assert s2 == warp.settings_from_record(project["warp_settings"]) and not new and "jolt.yaml" in how
    write_warp_settings(tmp_path / "other.settings", 1.368, (4092, 5760, 2000))
    s3, how, new = warp.resolve_project_settings(project, tmp_path / "other.settings")  # --warp_settings wins
    assert s3.tomo_dims_xyz == (4092, 5760, 2000) and new
    with pytest.raises(ValueError, match="--warp_settings"):  # disagreeing files and nothing recorded
        warp.resolve_project_settings({"warp_dir": str(tmp_path)})


def test_settings_volume_check():
    s = warp.WarpSettings(path=Path("w.settings"), pixel_size=1.368, tomo_dims_xyz=(4092, 5760, 2800))
    assert warp.settings_volume_check(s, BMP6_TOMO, 10.0) is None  # 559.8 x 787.97 x 383.04 vs 560 x 788 x 384
    assert "check that the settings belong" in warp.settings_volume_check(s, (560, 788, 300), 10.0)


def test_frame_series_settings_are_skipped_and_refused(tmp_path):
    """A WarpTools project has warp_frameseries.settings (movie extension, unused Tomo defaults) next to the
    tilt-series settings; EMPIAR-10499 has 3712 x 3712 x 1400 there and 3838 x 3710 x 2000 in warp_tiltseries."""
    write_warp_settings(tmp_path / "warp_tiltseries.settings", 1.7005, (3838, 3710, 2000))
    write_warp_settings(tmp_path / "warp_frameseries.settings", 1.7005, (3712, 3712, 1400), extension="*.tif")
    s, why = warp.find_settings(tmp_path)
    assert s.path.name == "warp_tiltseries.settings" and s.tomo_dims_xyz == (3838, 3710, 2000)
    assert "skipped frame-series warp_frameseries.settings" in why
    with pytest.raises(ValueError, match="frame-series settings"):
        warp.read_settings(tmp_path / "warp_frameseries.settings")
    (tmp_path / "warp_tiltseries.settings").unlink()
    s, why = warp.find_settings(tmp_path)
    assert s is None and "no tilt-series" in why
