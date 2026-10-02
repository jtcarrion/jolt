"""jolt import -> joint_optimize -> filter on a small synthetic Warp project (CPU)."""

import json
import subprocess
import sys

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.spatial import cKDTree

from tests.synthetic import synthetic_picks

TOMO_XYZ = (256, 256, 60)


def make_project(root):
    import warpxmlfile

    pts, picks, tilts = synthetic_picks(n_particles=20, image_hw=(256, 256), depth=60)
    stack_dir = root / "warp_tiltseries" / "tiltstack" / "TS_1"
    stack_dir.mkdir(parents=True)
    with mrcfile.new(stack_dir / "TS_1.st") as mrc:
        mrc.set_data(np.zeros((len(tilts), 256, 256), dtype=np.int8))
        mrc.voxel_size = 10.0
    (stack_dir / "TS_1.tlt").write_text("".join(f"{t:.2f}\n" for t in tilts))
    (stack_dir / "TS_1.xf").write_text("1.0 0.0 0.0 1.0 0.0 0.0\n" * len(tilts))
    rows = [(f"TS_1_{i + 1:02d}", x, y, s) for i, t in enumerate(tilts) for x, y, s in picks[float(t)]]
    table = pd.DataFrame(
        rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
    )
    starfile.write(table, stack_dir / "TS_1_picks.star")
    n = len(tilts)
    xml = warpxmlfile.WarpTiltSeries(
        angles=np.asarray(tilts, dtype=float),
        use_tilt=np.ones(n, dtype=bool),
        axis_angle=np.zeros(n),
        axis_offset_x=np.zeros(n),
        axis_offset_y=np.zeros(n),
    )
    warpxmlfile.write(xml, root / "warp_tiltseries" / "TS_1.xml")
    return pts


def jolt(*args):
    result = subprocess.run([sys.executable, "-m", "jolt", *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_import_joint_optimize_filter(tmp_path):
    true_center = make_project(tmp_path / "warp")
    project = tmp_path / "project"
    jolt(
        "import",
        "--warp_dir",
        tmp_path / "warp",
        "--project_dir",
        project,
        "--tomo_dims_xyz",
        ",".join(map(str, TOMO_XYZ)),
        "--picks_2d",
        "warp_tiltseries/tiltstack/{ts}/{ts}_picks.star",
    )
    jolt("joint_optimize", project, "--gpus", "cpu")

    out = project / "joint_optimize" / "TS_1"
    status = json.loads((out / "run_status.json").read_text())
    assert status["ba"]["status"] == "ok"
    particles = starfile.read(out / "TS_1_particles.star")
    xyz = particles[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
    # Expected corner-frame positions in the STAR (y down) frame.
    x, y, z = TOMO_XYZ
    expected = true_center + np.array([(x - 1) / 2, (y - 1) / 2, (z - 1) / 2])
    expected[:, 1] = (y - 1) - expected[:, 1]
    d, _ = cKDTree(expected).query(xyz)
    assert len(xyz) >= 18 and np.all(d < 1.5)
    assert (out / "TS_1_refined.xf").is_file() and (out / "TS_1_particles.mod").is_file()

    jolt("filter", project / "joint_optimize", "-o", tmp_path / "selected.star", "--min_fom", "0.5")
    summary = json.loads((tmp_path / "selected.json").read_text())
    assert summary["counts"]["input"] == len(xyz)
