# Benchmarks

Regression checks that need data or GPUs and therefore do not run in CI.

## CZI DS-10441 regression check

The benchmark of record is the CZI DS-10441 synthetic cohort: 27 tilt series with pretrained Topaz picks
("zero-shot") and 20 test tilt series with fine-tuned Topaz picks. JOLT output is scored with the CZI
`score_picks.py sweep3d-exclusive --distance-multiplier 1.0` at beta = 4 and beta = 1.

Reference results (code tag `jolt-reference-v0` of the original `particle_picker` repository, NVIDIA A100):

| cohort | F-beta4 | F-beta1 | TP |
|---|---|---|---|
| zero-shot, 27 series | 0.6280 | 0.3399 | 7766 |
| fine-tuned, 20 series | 0.7673 | 0.5902 | 6627 |

Bundle-adjustment results depend on the GPU type; compare against the reference on the same hardware (A100).

## `compare_to_reference.py`

Compares a JOLT run against a reference run per tilt series: the 3D coordinate sets (rounded, so row order
cannot cause false differences), the figure-of-merit column, and the refined `.xf`.

```bash
python benchmarks/compare_to_reference.py --new <run_dir> --ref <reference_run_dir> \
    --ts 5,8,9,14,25,26 --out report.json
```

`--ts 5,8,9,14,25,26` is the 6-series smoke set; run it before launching a full cohort.
