# Test case: CZI DS-10441 tilt series TS_5 (Warp-shaped)

A single synthetic tilt series laid out like a WarpTools project, so that `jolt import --warp_dir
tests/data/czi_TS_5` can read it. It is the test case for the `jolt` commands; the full CZI regression
check (27 + 20 tilt series) lives in `benchmarks/`.

```
warp_tiltseries/
├── TS_5.xml                         minimal Warp XML (see below)
└── tiltstack/TS_5/
    ├── TS_5.st                      PLACEHOLDER raw stack: correct header, all-zero pixels
    ├── TS_5.xf                      IMOD alignment (identity for every tilt in this data set)
    ├── TS_5.tlt                     tilt angles, -45 to +45 deg in 3 deg steps (31 tilts, ascending)
    └── TS_5_topaz_picks.star        2D picks: pretrained Topaz r8, preprocessed, top-quartile threshold
```

| File | Source | Notes |
|---|---|---|
| `TS_5.xf`, `TS_5.tlt` | CZI portal run `RN_16492_TS_5/tilt_series/tiltseries_16135/alignment/` | byte-identical copies |
| `TS_5_topaz_picks.star` | `CZI_syn_data/topaz_2d_output/topaz_r8_pp_full27/TS_5/topaz_r8_pp_full27_q25.star` | byte-identical copy; tilts are named `TS_01` ... `TS_31` (1-based stack index) |
| `TS_5.st` | header of `RN_16492_TS_5/.../TS_5.mrc` (630 x 630 x 31, 10 A/px) | pixels are zeros (8-bit), so git stores it in ~12 KB. JOLT reads only the header (raw image size). Do not use it for 2D picking. |
| `TS_5.xml` | written with `warpxmlfile` | angles from `TS_5.tlt`, all tilts used, axis angle and axis offsets 0 (what WarpTools derives from an identity `.xf`) |

Pixel coordinates are at the tilt-stack pixel size (10 A/px). The picks are in the raw-stack frame (raw and
aligned frames coincide here because the `.xf` is identity). Tomogram dimensions to pass:
`--tomo_dims_xyz 630,630,200`.

Reference output: the published JOLT run `CZI_syn_data/jolt_output/perturb_robustness_reg0p001/z_ba_truth/TS_5`
(code tag `jolt-reference-v0` of the `particle_picker` repository).
