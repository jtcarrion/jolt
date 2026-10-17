# Test case: CZI DS-10441 tilt series TS_5 (Warp-shaped)

A single synthetic tilt series laid out like a WarpTools project, so that `jolt import --warp_dir
tests/data/czi_TS_5` can read it. It is the test case for the `jolt` commands (see `benchmarks/README.md`).

```
warp_tiltseries/
├── TS_5.xml                         minimal Warp XML (see below)
└── tiltstack/TS_5/
    ├── TS_5.st                      PLACEHOLDER raw stack: correct header, all-zero pixels
    ├── TS_5.xf                      IMOD alignment (identity for every tilt in this data set)
    ├── TS_5.tlt                     tilt angles, -45 to +45 deg in 3 deg steps (31 tilts, ascending)
    └── TS_5_topaz_picks.star        2D picks: pretrained Topaz, radius 8, preprocessed, top-quartile threshold
```

| File | Source | Notes |
|---|---|---|
| `TS_5.xf`, `TS_5.tlt` | DS-10441, run TS_5, alignment files | byte-identical copies |
| `TS_5_topaz_picks.star` | pretrained Topaz (`resnet16`) on the DS-10441 TS_5 tilt images, extraction radius 8, top-quartile threshold | tilts are named `TS_01` ... `TS_31` (1-based stack index) |
| `TS_5.st` | header of the DS-10441 TS_5 tilt series (630 x 630 x 31, 10 A/px) | pixels are zeros (8-bit), so git stores it in ~12 KB. JOLT reads only the header (raw image size). Do not use it for 2D picking. |
| `TS_5.xml` | written with `warpxmlfile` | angles from `TS_5.tlt`, all tilts used, axis angle and axis offsets 0 (what WarpTools derives from an identity `.xf`) |

Pixel coordinates are at the tilt-stack pixel size (10 A/px). The picks are in the raw-stack frame (raw and
aligned frames coincide here because the `.xf` is identity). Tomogram dimensions to pass:
`--tomo_dims_xyz 630,630,200`.

## Data source

The tilt-series metadata come from dataset DS-10441, "CZII - CryoET Object Identification Challenge - Simulated
Training Data" (J. Schwartz, K. I. S. Harrington, M. Paraan), on the CryoET Data Portal of the Chan Zuckerberg Initiative
(https://cryoetdataportal.czscience.com/datasets/10441). See the dataset page for its license and the
portal's citation guidance.
