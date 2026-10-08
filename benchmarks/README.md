# Benchmark: the TS_5 test case

A single regression check on the bundled test case `tests/data/czi_TS_5` (CZI DS-10441 tilt series TS_5,
pretrained Topaz picks). It needs a GPU and is not run in CI.

```bash
jolt import --warp_dir tests/data/czi_TS_5 --project_dir ts5_check --tomo_dims_xyz 630,630,200 \
    --picks_2d "warp_tiltseries/tiltstack/{ts}/{ts}_topaz_picks.star"
jolt joint_optimize ts5_check --output_frame teamtomo   # CZI tomograms and ground truth share TeamTomo's hand
python benchmarks/compare_to_reference.py --new ts5_check/joint_optimize --ref <reference_run> --ts 5 \
    --new_star_name "{ts}_particles.star" --new_xf_name "{ts}_refined.xf" --cols rlnAutopickFigureOfMerit
```

`compare_to_reference.py` compares, per tilt series, the 3D coordinate sets (rounded, so row order cannot cause
false differences), the figure-of-merit column and the refined `.xf`.

Bundle-adjustment results depend slightly on the GPU type and on library versions, so compare runs made on
the same hardware and environment.
