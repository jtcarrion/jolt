#!/usr/bin/env python
"""Build a WarpTools-shaped view of the CZI DS-10441 cohort so that ``jolt import`` can read it.

The CZI portal layout (``RN_*_TS_<n>/tilt_series/tiltseries_*/TS_<n>.mrc`` plus ``alignment/TS_<n>.{xf,tlt}``)
has no Warp ``.xml``. This writes, for every run directory::

    <out>/warp_tiltseries/TS_<n>.xml                    minimal Warp XML (angles from the .tlt, all tilts used)
    <out>/warp_tiltseries/tiltstack/TS_<n>/TS_<n>.st    symbolic link to TS_<n>.mrc
    <out>/warp_tiltseries/tiltstack/TS_<n>/TS_<n>.xf    symbolic link
    <out>/warp_tiltseries/tiltstack/TS_<n>/TS_<n>.tlt   symbolic link

Usage::

    python benchmarks/make_czi_warp_view.py --czi_root <CZI_syn_data> --out <view_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import warpxmlfile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--czi_root", required=True, type=Path, help="directory holding the RN_*_TS_<n> run directories")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    runs = sorted(a.czi_root.glob("RN_*_TS_*"))
    xml_dir = a.out / "warp_tiltseries"
    for run in runs:
        name = "TS_" + run.name.rsplit("_TS_", 1)[1]
        series_dirs = sorted((run / "tilt_series").glob("tiltseries_*"))
        if len(series_dirs) != 1:
            raise SystemExit(f"{run}: expected one tilt_series/tiltseries_* directory, found {len(series_dirs)}")
        src = series_dirs[0]
        files = {
            ".st": src / f"{name}.mrc",
            ".xf": src / "alignment" / f"{name}.xf",
            ".tlt": src / "alignment" / f"{name}.tlt",
        }
        missing = [str(f) for f in files.values() if not f.is_file()]
        if missing:
            raise SystemExit(f"{run}: missing {missing}")
        stack_dir = xml_dir / "tiltstack" / name
        stack_dir.mkdir(parents=True, exist_ok=True)
        for suffix, target in files.items():
            link = stack_dir / f"{name}{suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.resolve())
        angles = np.array([float(x) for x in files[".tlt"].read_text().split()], dtype=float)
        n = len(angles)
        xml = warpxmlfile.WarpTiltSeries(
            angles=angles,
            use_tilt=np.ones(n, bool),
            axis_angle=np.zeros(n),
            axis_offset_x=np.zeros(n),
            axis_offset_y=np.zeros(n),
        )
        warpxmlfile.write(xml, xml_dir / f"{name}.xml")
    print(f"{len(runs)} tilt series -> {xml_dir}")


if __name__ == "__main__":
    main()
