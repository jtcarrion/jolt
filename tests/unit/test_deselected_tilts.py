"""Tilt images deselected in Warp (UseTilt false) are found at import and ignored by joint_optimize."""

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from jolt.io import config, warp
from jolt.pipeline import load_picks

ANGLES = [-6.0, -3.0, 0.0, 3.0, 6.0]
IDENTITY = [{"A11": 1.0, "A12": 0.0, "A21": 0.0, "A22": 1.0, "DX": 0.0, "DY": 0.0, "TILT": t} for t in ANGLES]


def _find(stack, xml_angles, use):
    return warp.deselected_stack_tilts("TS", "TS.xml", stack, np.asarray(xml_angles, float), np.asarray(use, bool))


def test_every_image_used():
    assert _find(ANGLES, ANGLES, [True] * 5) == ([], [])


def test_deselected_images_in_stack_order():
    assert _find(ANGLES, ANGLES, [True, False, True, True, False])[0] == [2, 5]


def test_stack_holding_only_used_tilts():
    # The .xml has two extra (unused) tilts that were never written to the stack: nothing is deselected.
    xml = [-12.0, -9.0] + ANGLES
    assert _find(ANGLES, xml, [False, False] + [True] * 5) == ([], [])


def test_xml_in_another_order_is_matched_by_angle():
    xml = ANGLES[::-1]  # 6, 3, 0, -3, -6
    use = [True, True, True, False, True]  # -3 deg is deselected
    assert _find(ANGLES, xml, use)[0] == [2]


def test_unmatched_angles_with_a_deselection_are_an_error():
    xml = [a + 2.0 for a in ANGLES]  # cannot tell which stack image the deselected entry is
    with pytest.raises(ValueError, match="cannot be identified"):
        _find(ANGLES, xml, [True, False, True, True, True])


def test_unmatched_angles_without_a_deselection_only_warn():
    deselected, warnings = _find(ANGLES, [a + 2.0 for a in ANGLES], [True] * 5)
    assert deselected == [] and warnings and "differ by up to 2.00 deg" in warnings[0]


def test_everything_deselected_is_an_error():
    with pytest.raises(ValueError, match="every tilt"):
        _find(ANGLES, ANGLES, [False] * 5)


def test_neither_layout_is_an_error():
    with pytest.raises(ValueError, match="lists 6 tilts"):
        _find(ANGLES, ANGLES + [9.0], [True, True, True, True, False, False])


def _write_series(root, use):
    import warpxmlfile

    folder = root / "warp_tiltseries" / "tiltstack" / "TS_1"
    folder.mkdir(parents=True)
    with mrcfile.new(folder / "TS_1.st") as mrc:
        mrc.set_data(np.zeros((len(ANGLES), 8, 8), dtype=np.int8))
        mrc.voxel_size = 10.0
    (folder / "TS_1.tlt").write_text("".join(f"{a:.2f}\n" for a in ANGLES))
    (folder / "TS_1.xf").write_text("1.0 0.0 0.0 1.0 0.0 0.0\n" * len(ANGLES))
    n = len(ANGLES)
    xml = warpxmlfile.WarpTiltSeries(
        angles=np.asarray(ANGLES, float),
        use_tilt=np.asarray(use, bool),
        axis_angle=np.zeros(n),
        axis_offset_x=np.zeros(n),
        axis_offset_y=np.zeros(n),
    )
    warpxmlfile.write(xml, root / "warp_tiltseries" / "TS_1.xml")


def test_discover_reports_deselected_tilts(tmp_path):
    _write_series(tmp_path, [True, True, False, True, True])
    (series,) = warp.discover(tmp_path)
    assert series.n_tilts == 5 and series.deselected_tilts == [3]


def test_load_picks_drops_deselected_tilts(tmp_path):
    rows = [(f"TS_1_{i + 1:02d}", 10.0 + i, 20.0, 0.5) for i in range(len(ANGLES))]
    path = tmp_path / "picks.star"
    starfile.write(
        pd.DataFrame(
            rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
        ),
        path,
    )
    picks = load_picks(path, ANGLES, "aligned", IDENTITY, (64, 64), (64, 64), deselected=(2, 5))
    assert sorted(picks) == [-6.0, 0.0, 3.0]
    assert load_picks(path, ANGLES, "aligned", IDENTITY, (64, 64), (64, 64)).keys() == set(ANGLES)


def test_config_validates_deselected_tilts(tmp_path):
    cfg = {
        "config_version": config.CONFIG_VERSION,
        "project": {"tomo_dims_xyz": [64, 64, 20]},
        "tilt_series": {
            "TS_1": {"raw_stack": "a", "xf": "b", "tlt": "c", "n_tilts": 5, "raw_dims_xy": [64, 64], "pixel_size": 1.0}
        },
    }
    config.validate(cfg)  # no key: older projects keep working
    cfg["tilt_series"]["TS_1"]["deselected_tilts"] = [2, 5]
    config.validate(cfg)
    cfg["tilt_series"]["TS_1"]["deselected_tilts"] = [6]
    with pytest.raises(ValueError, match="deselected_tilts"):
        config.validate(cfg)


def test_finetune_skips_deselected_tilt_images(tmp_path):
    import subprocess
    import sys

    import warpxmlfile

    warp_dir = tmp_path / "warp"
    _write_series(warp_dir, [True, False, True, True, True])  # TS_1: stack position 2 deselected
    second = warp_dir / "warp_tiltseries" / "tiltstack" / "TS_2"
    second.mkdir()
    for suffix in (".st", ".tlt", ".xf"):
        (second / f"TS_2{suffix}").write_bytes(
            (warp_dir / "warp_tiltseries/tiltstack/TS_1" / f"TS_1{suffix}").read_bytes()
        )
    xml = warpxmlfile.read(warp_dir / "warp_tiltseries" / "TS_1.xml")
    xml.use_tilt = np.ones(len(ANGLES), dtype=bool)
    warpxmlfile.write(xml, warp_dir / "warp_tiltseries" / "TS_2.xml")
    for name in ("TS_1", "TS_2"):
        rows = [(f"{name}_{i + 1:02d}", 4.0, 4.0, 1.0) for i in range(len(ANGLES))]
        starfile.write(
            pd.DataFrame(
                rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
            ),
            tmp_path / f"{name}_labels.star",
        )
    project = tmp_path / "project"

    def jolt(*args):
        result = subprocess.run([sys.executable, "-m", "jolt", *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    jolt("import", "--warp_dir", warp_dir, "--project_dir", project, "--tomo_dims_xyz", "8,8,4")
    jolt("finetune", project, "--labels_2d", str(tmp_path / "{ts}_labels.star"), "--heldout_ts", "TS_2", "--dry_run")
    train = (project / "finetune" / "data" / "train_images.txt").read_text().split()
    test = (project / "finetune" / "data" / "test_images.txt").read_text().split()
    train_names = {w for w in train if w.startswith("TS_1_")}
    assert train_names == {f"TS_1_tilt_{i:02d}" for i in (1, 3, 4, 5)}
    assert {w for w in test if w.startswith("TS_2_")} == {f"TS_2_tilt_{i:02d}" for i in range(1, 6)}


def test_warp_alignment_files(tmp_path):
    from jolt.pipeline import write_warp_alignment

    tlt = tmp_path / "in.tlt"
    tlt.write_text("  -6.00\n -3.0\n   0.00\n  3.00\n6\n\n")  # odd formatting and a trailing blank line
    rows = [dict(r, DX=float(i), DY=-float(i)) for i, r in enumerate(IDENTITY)]
    info = write_warp_alignment(tmp_path, "TS_1", rows, tlt, deselected=(2, 5))
    assert info == {"xf": "TS_1.xf", "tlt": "TS_1.tlt", "stack_positions": [1, 3, 4], "warp_in_use_tilts": None}
    assert (tmp_path / "TS_1.tlt").read_text() == "  -6.00\n   0.00\n  3.00\n"  # verbatim lines, deselected left out
    xf = np.loadtxt(tmp_path / "TS_1.xf")
    np.testing.assert_allclose(xf[:, 4], [0.0, 2.0, 3.0])
    write_warp_alignment(tmp_path, "TS_2", rows, tlt)
    assert len((tmp_path / "TS_2.tlt").read_text().splitlines()) == 5
    with pytest.raises(ValueError, match="tilt angles for 4"):
        write_warp_alignment(tmp_path, "TS_3", rows[:4], tlt)


def test_warp_alignment_row_count_must_match_warp(tmp_path):
    from jolt.pipeline import write_warp_alignment

    tlt = tmp_path / "in.tlt"
    tlt.write_text("".join(f"{a}\n" for a in ANGLES))
    info = write_warp_alignment(tmp_path, "TS_1", IDENTITY, tlt, deselected=(2,), expected_rows=4)
    assert info["warp_in_use_tilts"] == 4 and len(info["stack_positions"]) == 4
    with pytest.raises(ValueError, match="4 alignment rows but Warp has 5 tilts in use"):
        write_warp_alignment(tmp_path, "TS_2", IDENTITY, tlt, deselected=(2,), expected_rows=5)


def test_current_deselection_counts_and_guard(tmp_path, monkeypatch):
    import warpxmlfile

    from jolt.commands.joint_optimize import current_deselection
    from jolt.io import warp as warp_module

    _write_series(tmp_path, [True, False, True, True, True])
    folder = tmp_path / "warp_tiltseries"
    entry = {"xml": str(folder / "TS_1.xml"), "tlt": str(folder / "tiltstack/TS_1/TS_1.tlt"), "deselected_tilts": [2]}
    assert current_deselection("TS_1", entry) == ((2,), 4, None)

    # A stack written with only the used tilts: the XML has two extra unused entries; nothing is removed twice.
    xml = warpxmlfile.read(folder / "TS_1.xml")
    xml.angles = np.asarray([-12.0, -9.0] + ANGLES, dtype=float)
    xml.use_tilt = np.asarray([False, False] + [True] * 5, dtype=bool)
    for key in ("axis_angle", "axis_offset_x", "axis_offset_y"):
        setattr(xml, key, np.zeros(7))
    warpxmlfile.write(xml, folder / "TS_1.xml")
    deselected, in_use, warning = current_deselection("TS_1", {**entry, "deselected_tilts": []})
    assert deselected == () and in_use == 5 and warning is None

    # Defensive guard: rows that would not match Warp's in-use count stop the run before anything is computed.
    monkeypatch.setattr(warp_module, "deselected_stack_tilts", lambda *a: ([1], []))
    with pytest.raises(SystemExit, match="Warp could not import"):
        current_deselection("TS_1", entry)
