"""jolt finetune: 3D labels give the same 2D labels whichever output frame they are given in."""

import numpy as np
import pandas as pd

from jolt import geometry
from jolt.commands.finetune import labels_from_3d

TOMO = (652, 630, 340)


def _entry(tmp_path):
    tilts = np.arange(60.0, -60.1, -3.0)  # descending, like EMPIAR-10499
    (tmp_path / "s.tlt").write_text("".join(f"{t:.2f}\n" for t in tilts))
    a = np.deg2rad(4.3)
    row = f"{np.cos(a):.7f} {-np.sin(a):.7f} {np.sin(a):.7f} {np.cos(a):.7f} 3.5 -2.0\n"
    (tmp_path / "s.xf").write_text(row * len(tilts))
    return {"tlt": str(tmp_path / "s.tlt"), "xf": str(tmp_path / "s.xf"), "raw_dims_xy": [652, 630]}


def test_labels_are_identical_in_every_frame(tmp_path):
    entry = _entry(tmp_path)
    rng = np.random.default_rng(3)
    jolt_xyz = np.column_stack([rng.uniform(60, 590, 40), rng.uniform(60, 570, 40), rng.uniform(40, 300, 40)])
    cols = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    reference = labels_from_3d(pd.DataFrame(jolt_xyz, columns=cols), entry, TOMO)
    for frame in geometry.OUTPUT_FRAMES:
        f = geometry.output_frame_transform(
            frame,
            TOMO,
            10.0,
            warp_image_dims_angstrom=(6526.519, 6308.855),
            warp_volume_dims_angstrom=(6526.519, 6308.855, 3401.0),
            image_rotation_deg=4.3,
        )
        got = labels_from_3d(pd.DataFrame(f.to_frame(jolt_xyz), columns=cols), entry, TOMO, f)
        assert got.keys() == reference.keys()
        for k in reference:
            np.testing.assert_allclose(got[k], reference[k], atol=1e-6)
