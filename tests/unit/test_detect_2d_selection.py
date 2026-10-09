"""detect_2d selection: the best 1200 picks per tilt image by default; --threshold replaces it; never both."""

import subprocess
import sys

import mrcfile
import numpy as np
import pytest

from jolt.commands.detect_2d import pick_selection


def test_pick_selection():
    assert pick_selection(None, None) == (None, 1200)  # default: a per-tilt budget
    assert pick_selection(None, 300) == (None, 300)
    assert pick_selection(-3.0, None) == (-3.0, None)  # a threshold means no top-K cut
    with pytest.raises(SystemExit, match="not both"):
        pick_selection(-3.0, 300)
    with pytest.raises(SystemExit, match="positive"):
        pick_selection(None, 0)


def _project(tmp_path):
    import warpxmlfile

    angles = [-3.0, 0.0, 3.0]
    folder = tmp_path / "warp" / "warp_tiltseries" / "tiltstack" / "TS_1"
    folder.mkdir(parents=True)
    with mrcfile.new(folder / "TS_1.st") as mrc:
        mrc.set_data(np.zeros((3, 8, 8), dtype=np.int8))
        mrc.voxel_size = 10.0
    (folder / "TS_1.tlt").write_text("".join(f"{a}\n" for a in angles))
    (folder / "TS_1.xf").write_text("1.0 0.0 0.0 1.0 0.0 0.0\n" * 3)
    xml = warpxmlfile.WarpTiltSeries(
        angles=np.asarray(angles),
        use_tilt=np.ones(3, bool),
        axis_angle=np.zeros(3),
        axis_offset_x=np.zeros(3),
        axis_offset_y=np.zeros(3),
    )
    warpxmlfile.write(xml, tmp_path / "warp" / "warp_tiltseries" / "TS_1.xml")
    project = tmp_path / "project"
    run("import", "--warp_dir", tmp_path / "warp", "--project_dir", project, "--tomo_dims_xyz", "8,8,4")
    return project


def run(*args, ok=True):
    result = subprocess.run([sys.executable, "-m", "jolt", *map(str, args)], capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_detect_2d_cli_selection(tmp_path):
    project = _project(tmp_path)
    base = ("detect_2d", project, "--radius", "6", "--dry_run")
    assert "top_k=1200" in run(*base)
    assert "top_k=300" in run(*base, "--top_k", "300")
    out = run(*base, "--threshold", "-3")
    assert "threshold=-3.0" in out and "top_k" not in out
    assert "not allowed with argument" in run(*base, "--threshold", "-3", "--top_k", "300", ok=False)
