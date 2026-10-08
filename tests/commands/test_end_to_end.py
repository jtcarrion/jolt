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
    jolt("joint_optimize", project, "--gpus", "cpu")  # default output frame: warp
    jolt("joint_optimize", project, "--gpus", "cpu", "--output_frame", "teamtomo", "--output_dir", project / "tt")

    # Expected positions in JOLT's own corner frame (STAR, y down).
    x, y, z = TOMO_XYZ
    expected = true_center + np.array([(x - 1) / 2, (y - 1) / 2, (z - 1) / 2])
    expected[:, 1] = (y - 1) - expected[:, 1]
    # warp: this synthetic XML has no physical dimensions, so exact binning: z -> NZ - 0.5 - z.
    # teamtomo: volume center NZ // 2, so z + 0.5 for the even depth used here.
    in_warp = expected.copy()
    in_warp[:, 2] = z - 0.5 - expected[:, 2]
    in_teamtomo = expected + np.array([0.0, 0.0, 0.5])
    for out, frame, truth in (
        (project / "joint_optimize" / "TS_1", "warp", in_warp),
        (project / "tt" / "TS_1", "teamtomo", in_teamtomo),
    ):
        status = json.loads((out / "run_status.json").read_text())
        assert status["ba"]["status"] == "ok"
        assert status["output_frame"]["name"] == frame
        particles = starfile.read(out / "TS_1_particles.star")
        xyz = particles[["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]].to_numpy(float)
        d, _ = cKDTree(truth).query(xyz)
        assert len(xyz) >= 18 and np.all(d < 1.5), frame
        assert (out / "TS_1_refined.xf").is_file() and (out / "TS_1_particles.mod").is_file()
    warp_xyz = starfile.read(project / "joint_optimize/TS_1/TS_1_particles.star")
    tt_xyz = starfile.read(project / "tt/TS_1/TS_1_particles.star")
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    # Same particles, exactly mirrored in z between the frames: z_warp + z_teamtomo = NZ (= (NZ - 0.5) + 0.5).
    np.testing.assert_allclose(warp_xyz[cols[:2]].to_numpy(float), tt_xyz[cols[:2]].to_numpy(float), atol=1e-6)
    np.testing.assert_allclose(warp_xyz["rlnCoordinateZ"] + tt_xyz["rlnCoordinateZ"], z, atol=1e-5)

    jolt("filter", project / "joint_optimize", "-o", tmp_path / "selected.star", "--min_fom", "0.5")
    summary = json.loads((tmp_path / "selected.json").read_text())
    assert summary["counts"]["input"] == len(xyz)
