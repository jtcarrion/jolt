#!/usr/bin/env python
"""Compare a JOLT run against a reference run, per tilt series, exactly.

For each TS it compares BA_residual_filtered_3d.star (rounded coordinate multisets, so row order
and n_jobs cannot cause false differences; unrounded max nearest-neighbor distance as a
sanity check), the FOM column, and the refined <TS>.xf. Writes a JSON report and prints a table.

Usage:
  compare_to_reference.py --new <run_dir> --ref <run_dir> --ts 5,8,9,14,25,26 --out report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import starfile
from scipy.spatial import cKDTree


def _read_points(star_path):
    df = starfile.read(star_path, always_dict=True)
    df = next(iter(df.values()))
    cols = {c.lstrip("_"): c for c in df.columns}
    xyz = df[[cols["rlnCoordinateX"], cols["rlnCoordinateY"], cols["rlnCoordinateZ"]]].to_numpy(float)
    fom_col = cols.get("rlnAutopickFigureOfMerit")
    fom = df[fom_col].to_numpy(float) if fom_col else None
    extra = {
        k: df[c].to_numpy(float)
        for k, c in cols.items()
        if k not in ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ") and np.issubdtype(df[c].dtype, np.number)
    }
    return xyz, fom, extra


def _read_xf(path):
    rows = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    return np.array(rows, dtype=float)


def _col_diff(a, b):
    """Max |a-b| with NaN-awareness; returns 'nan-mismatch' if NaN positions differ."""
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return "nan-mismatch"
    return float(np.max(np.abs(a[~na] - b[~nb]))) if (~na).any() else 0.0


def compare_ts(new_dir, ref_dir, ts, decimals, extra_cols=()):
    r = {"ts": ts}
    new_star, ref_star = new_dir / "BA_residual_filtered_3d.star", ref_dir / "BA_residual_filtered_3d.star"
    if not new_star.exists() or not ref_star.exists():
        r["status"] = f"missing star (new={new_star.exists()}, ref={ref_star.exists()})"
        return r
    xn, fn, en = _read_points(new_star)
    xr, fr, er = _read_points(ref_star)
    cn = Counter(map(tuple, np.round(xn, decimals)))
    cr = Counter(map(tuple, np.round(xr, decimals)))
    common = sum((cn & cr).values())
    r.update(n_new=len(xn), n_ref=len(xr), n_common_rounded=common, coords_identical=(cn == cr))
    if len(xn) and len(xr):
        d_nr, _ = cKDTree(xr).query(xn)
        d_rn, _ = cKDTree(xn).query(xr)
        r["max_nn_dist_px"] = float(max(d_nr.max(), d_rn.max()))
    if fn is not None and fr is not None and r["coords_identical"]:
        on, orr = np.lexsort(xn.T[::-1]), np.lexsort(xr.T[::-1])
        r["max_fom_diff"] = float(np.max(np.abs(fn[on] - fr[orr]))) if len(fn) else 0.0
        for c in extra_cols:
            if c in en and c in er:
                r[f"max_diff_{c}"] = _col_diff(en[c][on], er[c][orr])
            else:
                r[f"max_diff_{c}"] = f"absent (new={c in en}, ref={c in er})"
    for c in extra_cols:
        r.setdefault(f"present_{c}", c in en)
    xf_new, xf_ref = new_dir / f"{ts}.xf", ref_dir / f"{ts}.xf"
    if xf_new.exists() and xf_ref.exists():
        a, b = _read_xf(xf_new), _read_xf(xf_ref)
        r["xf_max_abs_diff"] = float(np.max(np.abs(a - b))) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
    r["status"] = "ok"
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--ref", required=True, type=Path)
    p.add_argument("--ts", required=True, help="comma-separated numbers or TS_ names")
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--cols", default="", help="extra numeric STAR columns to compare, comma-separated")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    names = [t if t.startswith("TS_") else f"TS_{t}" for t in a.ts.split(",")]
    cols = [c for c in a.cols.split(",") if c]
    rows = [compare_ts(a.new / t, a.ref / t, t, a.decimals, cols) for t in names]
    all_same = all(r.get("coords_identical") for r in rows)
    print(f"{'TS':<7}{'n_new':>7}{'n_ref':>7}{'common':>8}{'same':>6}{'maxNN':>9}{'maxFOMd':>10}{'xf_maxd':>10}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['ts']:<7} {r['status']}")
            continue
        print(
            f"{r['ts']:<7}{r['n_new']:>7}{r['n_ref']:>7}{r['n_common_rounded']:>8}"
            f"{str(r['coords_identical']):>6}{r.get('max_nn_dist_px', float('nan')):>9.4f}"
            f"{str(round(r['max_fom_diff'], 6)) if 'max_fom_diff' in r else '-':>10}"
            f"{str(r.get('xf_max_abs_diff', '-'))[:9]:>10}"
        )
    for c in cols:
        print(f"  {c}: " + ", ".join(f"{r['ts']}={r.get('max_diff_' + c, '-')}" for r in rows))
    print(f"ALL COORDINATE SETS IDENTICAL: {all_same}")
    if a.out:
        a.out.write_text(
            json.dumps(
                {
                    "new": str(a.new),
                    "ref": str(a.ref),
                    "decimals": a.decimals,
                    "all_coords_identical": all_same,
                    "per_ts": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
