"""The bundled TS_5 test case must be internally consistent."""

from pathlib import Path

import mrcfile
import numpy as np

TS_DIR = Path(__file__).parents[1] / "data" / "czi_TS_5" / "warp_tiltseries"
STACK_DIR = TS_DIR / "tiltstack" / "TS_5"


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def test_tlt_xf_and_stack_have_the_same_number_of_tilts():
    tilts = read_rows(STACK_DIR / "TS_5.tlt")
    xf_rows = read_rows(STACK_DIR / "TS_5.xf")
    with mrcfile.open(STACK_DIR / "TS_5.st", header_only=True, permissive=True) as mrc:
        nx, ny, nz = (int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz))
    assert len(tilts) == len(xf_rows) == nz == 31
    assert (nx, ny) == (630, 630)
    assert all(len(row) == 6 for row in xf_rows)


def test_tilt_angles_are_ascending_minus_45_to_45():
    angles = np.array([float(row[0]) for row in read_rows(STACK_DIR / "TS_5.tlt")])
    np.testing.assert_allclose(angles, np.arange(-45, 46, 3), atol=1e-2)


def test_warp_xml_exists():
    assert (TS_DIR / "TS_5.xml").is_file()
