"""A tilt deselected in Warp behaves exactly like a tilt without picks, whatever picks it holds (CPU)."""

import json
import subprocess
import sys

import mrcfile
import numpy as np
import pandas as pd
import starfile

from tests.synthetic import synthetic_picks

TOMO_XYZ = (256, 256, 60)
DESELECTED = (3, 9)  # 1-based stack positions


def make_project(root, use_tilt, drop=(), garbage=()):
    """Synthetic Warp project; picks on ``drop`` positions are removed, on ``garbage`` positions replaced by noise."""
    import warpxmlfile

    _, picks, tilts = synthetic_picks(n_particles=20, image_hw=(256, 256), depth=60)
    stack_dir = root / "warp_tiltseries" / "tiltstack" / "TS_1"
    stack_dir.mkdir(parents=True)
    with mrcfile.new(stack_dir / "TS_1.st") as mrc:
        mrc.set_data(np.zeros((len(tilts), 256, 256), dtype=np.int8))
        mrc.voxel_size = 10.0
    (stack_dir / "TS_1.tlt").write_text("".join(f"{t:.2f}\n" for t in tilts))
    (stack_dir / "TS_1.xf").write_text(
        "".join(f"1.0 0.0 0.0 1.0 {0.3 * i:.3f} {-0.2 * i:.3f}\n" for i in range(len(tilts)))
    )
    rng = np.random.default_rng(7)
    rows = []
    for i, t in enumerate(tilts):
        if i + 1 in drop:
            continue
        if i + 1 in garbage:
            rows += [(f"TS_1_{i + 1:02d}", *rng.uniform(5, 250, 2), 0.9) for _ in range(40)]
        else:
            rows += [(f"TS_1_{i + 1:02d}", x, y, s) for x, y, s in picks[float(t)]]
    table = pd.DataFrame(
        rows, columns=["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnAutopickFigureOfMerit"]
    )
    starfile.write(table, stack_dir / "TS_1_picks.star")
    n = len(tilts)
    xml = warpxmlfile.WarpTiltSeries(
        angles=np.asarray(tilts, dtype=float),
        use_tilt=np.asarray(use_tilt(n), dtype=bool),
        axis_angle=np.zeros(n),
        axis_offset_x=np.zeros(n),
        axis_offset_y=np.zeros(n),
    )
    warpxmlfile.write(xml, root / "warp_tiltseries" / "TS_1.xml")
    return stack_dir


def jolt(*args):
    result = subprocess.run([sys.executable, "-m", "jolt", *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def run_arm(tmp_path, label, **kwargs):
    warp_dir = tmp_path / label / "warp"
    stack_dir = make_project(warp_dir, **kwargs)
    project = tmp_path / label / "project"
    imported = jolt(
        "import",
        "--warp_dir",
        warp_dir,
        "--project_dir",
        project,
        "--tomo_dims_xyz",
        ",".join(map(str, TOMO_XYZ)),
        "--picks_2d",
        "warp_tiltseries/tiltstack/{ts}/{ts}_picks.star",
    )
    jolt("joint_optimize", project, "--gpus", "cpu")
    out = project / "joint_optimize" / "TS_1"
    return stack_dir, out, imported.stdout


def body(path):
    """STAR file content without comment lines (the header holds a timestamp)."""
    return [line for line in path.read_text().splitlines() if not line.startswith("#")]


def deselect(n):
    return [i + 1 not in DESELECTED for i in range(n)]


def test_deselected_tilts_match_removed_picks(tmp_path):
    _, out_deselected, log = run_arm(tmp_path, "deselected", use_tilt=deselect)
    _, out_removed, _ = run_arm(tmp_path, "removed", use_tilt=lambda n: [True] * n, drop=DESELECTED)
    stack_dir, out_garbage, _ = run_arm(tmp_path, "garbage", use_tilt=deselect, garbage=DESELECTED)
    assert "2 tilt image(s) deselected in Warp" in log

    status = json.loads((out_deselected / "run_status.json").read_text())
    assert [d["stack_position"] for d in status["deselected_tilts"]] == list(DESELECTED)
    assert status["ba"]["status"] == "ok"
    assert "deselected_tilts" not in json.loads((out_removed / "run_status.json").read_text())

    # Particles are identical in all three arms; so are the refined rows of the used tilts.
    reference = body(out_deselected / "TS_1_particles.star")
    for out in (out_removed, out_garbage):
        assert body(out / "TS_1_particles.star") == reference
    assert (out_garbage / "TS_1.xf").read_text() == (out_deselected / "TS_1.xf").read_text()
    assert (out_garbage / "TS_1.tlt").read_text() == (out_deselected / "TS_1.tlt").read_text()
    assert len(starfile.read(out_deselected / "TS_1_particles.star")) >= 18

    # The Warp-ready .xf/.tlt hold the in-use tilts only (Warp maps rows onto UseTilt), in stack order; the .tlt
    # lines are the stack's own, verbatim. Without deselection every stack image has a row.
    original = np.loadtxt(stack_dir / "TS_1.xf")
    used = [i for i in range(len(original)) if i + 1 not in DESELECTED]
    refined = np.loadtxt(out_deselected / "TS_1.xf")
    assert refined.shape == (len(used), 6)
    np.testing.assert_array_equal(refined, np.loadtxt(out_removed / "TS_1.xf")[used])
    tlt_lines = [line for line in (stack_dir / "TS_1.tlt").read_text().splitlines() if line.strip()]
    assert (out_deselected / "TS_1.tlt").read_text().splitlines() == [tlt_lines[i] for i in used]
    assert (out_removed / "TS_1.tlt").read_text().splitlines() == tlt_lines
    assert status["warp_alignment"]["stack_positions"] == [i + 1 for i in used]
    assert status["warp_alignment"]["warp_in_use_tilts"] == len(used)
    assert np.abs(refined[:, 4:] - original[used, 4:]).max() > 0  # the used tilts were refined
    np.testing.assert_allclose(refined[:, :4], original[used, :4], atol=1e-6)  # rotation untouched
    assert not (out_deselected / "TS_1_refined.xf").exists()


def test_deselection_made_in_warp_after_import_is_honored(tmp_path):
    import warpxmlfile

    stack_dir, _, _ = run_arm(tmp_path, "late", use_tilt=lambda n: [True] * n)  # imported with nothing deselected
    xml_path = tmp_path / "late" / "warp" / "warp_tiltseries" / "TS_1.xml"
    xml = warpxmlfile.read(xml_path)
    xml.use_tilt = np.asarray(deselect(len(xml.use_tilt)), dtype=bool)  # the user deselects two tilts in Warp
    warpxmlfile.write(xml, xml_path)
    project = tmp_path / "late" / "project"
    jolt("joint_optimize", project, "--gpus", "cpu", "--output_dir", project / "again")
    assert "the Warp XML now deselects stack positions [3, 9]" in (project / "again" / "joint_optimize.log").read_text()
    out = project / "again" / "TS_1"
    status = json.loads((out / "run_status.json").read_text())
    assert [d["stack_position"] for d in status["deselected_tilts"]] == list(DESELECTED)
    _, out_deselected, _ = run_arm(tmp_path, "desel", use_tilt=deselect)
    assert body(out / "TS_1_particles.star") == body(out_deselected / "TS_1_particles.star")
    assert (out / "TS_1.xf").read_text() == (out_deselected / "TS_1.xf").read_text()
    assert (out / "TS_1.tlt").read_text() == (out_deselected / "TS_1.tlt").read_text()


def test_stack_already_trimmed_is_not_trimmed_again(tmp_path):
    """The XML lists two extra tilts that were never written to the stack: every stack row is kept."""
    import warpxmlfile

    _, out_plain, _ = run_arm(tmp_path, "plain", use_tilt=lambda n: [True] * n)
    warp_dir = tmp_path / "trimmed" / "warp"
    make_project(warp_dir, use_tilt=lambda n: [True] * n)
    xml_path = warp_dir / "warp_tiltseries" / "TS_1.xml"
    xml = warpxmlfile.read(xml_path)
    n = len(xml.angles)
    xml.angles = np.concatenate([[-70.0, -66.0], np.asarray(xml.angles, dtype=float)])
    xml.use_tilt = np.concatenate([[False, False], np.ones(n, dtype=bool)])
    for key in ("axis_angle", "axis_offset_x", "axis_offset_y"):
        setattr(xml, key, np.zeros(n + 2))
    warpxmlfile.write(xml, xml_path)
    project = tmp_path / "trimmed" / "project"
    imported = jolt(
        "import",
        "--warp_dir",
        warp_dir,
        "--project_dir",
        project,
        "--tomo_dims_xyz",
        ",".join(map(str, TOMO_XYZ)),
        "--picks_2d",
        "warp_tiltseries/tiltstack/{ts}/{ts}_picks.star",
    )
    assert "deselected in Warp" not in imported.stdout
    jolt("joint_optimize", project, "--gpus", "cpu")
    out = project / "joint_optimize" / "TS_1"
    status = json.loads((out / "run_status.json").read_text())
    assert status["warp_alignment"]["warp_in_use_tilts"] == n
    assert len(np.loadtxt(out / "TS_1.xf")) == n and len((out / "TS_1.tlt").read_text().splitlines()) == n
    assert (out / "TS_1.xf").read_text() == (out_plain / "TS_1.xf").read_text()
    assert body(out / "TS_1_particles.star") == body(out_plain / "TS_1_particles.star")
