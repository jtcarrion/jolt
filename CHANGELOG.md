# Changelog

All notable changes to JOLT are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/).
Entries marked **changes the output** alter the particles or the alignment JOLT writes for the same input.

## [Unreleased]

### Changed
- `finetune` defaults follow a benchmark on three in-situ ribosome datasets (EMPIAR-10499, an *E. coli* dataset and
  CZI DS-10301; 2D per-tilt recall and 3D JOLT recall on held-out series). **Changes the output** (the trained model):
  - 12 epochs x 1,000 minibatches (was 20 x 100). 2,000 steps under-trained small label sets: on *Chlamydomonas*
    the shorter schedule alone cost 5 points of 2D recall, and the old defaults as a whole 7 points of 3D JOLT
    recall at 50 A (0.545 vs 0.616).
  - `--units 64` (new option; Topaz's default pretrained `resnet16` has 32 units), `--learning_rate 2e-4` (was 1e-4).
  - `--num_particles` defaults to `--num_particles_factor` (new, 1.5) x the mean labels per training tilt image
    (was a fixed 350). `finetune` refuses a value at or below that mean: with it, Topaz silently replaced the
    GE-binomial objective by PN, treating unlabeled particles as background.
- `finetune --select` (new) chooses the weights `detect_2d` uses; every epoch's model is kept. The default,
  `average`, is the uniform weight average of the last quarter of the epochs (10-12 of 12, `model_average.sav`),
  instead of the final epoch: it matched or beat every single epoch on all three datasets and varied least between
  repeated runs (*Chlamydomonas* test 2D recall 0.493, vs 0.478 for the final epoch and 0.413 for the old
  defaults). `best_auprc` (the epoch with the best held-out AUPRC) and `last` are available; held-out AUPRC on a few
  tilt series is noisy and picked epoch 1 of 12 (0.434) in the same run. `jolt.yaml` also records the rule, the
  epochs used, the best-AUPRC epoch, the final epoch's weights, the held-out AUPRC per epoch, pi,
  `--num_particles`, the sampler and the Topaz version.
- `finetune` runs `topaz train` through `jolt/_topaz_train.py` in Topaz's interpreter, which seeds it as before and,
  for Topaz 0.3.18 (checked by version and source hash), replaces the positive-crop sampler with an O(1) draw from
  the same distribution. Stock Topaz's `DataFrame.sample()` slows down in proportion to the number of labels
  (0.5 s per step at ~41k labels, ~6 s at ~380k), which made 12,000 steps take most of a day on densely labeled
  data.

### Fixed
- `finetune` computes the Topaz class prior pi itself and passes it with `topaz train -n 0 --pi`. Topaz 0.3.18's
  `pixels_given_radius` counts a 3D grid for 2D images, which made pi (2r + 1) times too large (7x at radius 3).
  A Topaz run that switches to PN anyway now stops `finetune` with an error.

## [0.0.1_beta] - unreleased

### Added
- `joint_optimize` writes the refined alignment as `<TS>/<TS>.xf` plus `<TS>/<TS>.tlt` (the stack's own tilt angles,
  copied unchanged), so `WarpTools ts_import_alignments --alignments <project>/joint_optimize` reads the output
  folder directly. A new documentation page describes the round trip into Warp (import, `ts_ctf`,
  `ts_reconstruct`, `ts_export_particles`).
- Tilt images deselected in Warp (`UseTilt`) are ignored instead of the tilt series being refused; they are left
  out of the Warp-ready `.xf`/`.tlt` so that Warp keeps them deselected.
- Bundle-adjustment loss `--ba_loss evidence` (default): each particle is scored by the evidence of all picks
  within the match distance, with a pick localization scale estimated per tilt series (`--ba_sigma auto`;
  `--ba_gamma`). Smooth, stable under tiny perturbations, and bundle adjustment runs 3-5x faster.
  `--ba_loss nearest` keeps the previous loss. **Changes the output**: CZI DS-10441 F-beta4 / F-beta1
  0.6344 / 0.3405 -> 0.6405 / 0.3537 (zero-shot) and 0.7778 / 0.5917 -> 0.7791 / 0.604 (fine-tuned), with fewer
  false positives.
- `detect_2d` also writes its picks as an IMOD model, `<TS>_picks.mod`, for viewing on the raw stack in `3dmod`.
- `joint_optimize --iter N` (default 1) runs N passes of picks -> MLESAC -> bundle adjustment, each from the previous
  pass's refined alignment. The refined alignment is cumulative; `run_status.json` lists every pass.
- Gold-bead anchors: `jolt import --fiducials imod` registers etomo `.fid` bead tracks; `joint_optimize`
  (`--fiducials anchor`, `--fiducial_weight 1.0`) adds them to bundle adjustment with fixed correspondences and
  writes `<ts>_fiducials.star`.
- Output frames: `joint_optimize --output_frame {warp,teamtomo}` (default `warp`, the frame of WarpTools
  tomograms and of `ts_export_particles`, with Warp's sub-voxel centers and size-rounding scale read from the Warp
  XML) and `finetune --labels_frame`. The frame is recorded in `run_status.json`.
- Installation notes for Topaz with a CUDA PyTorch.

### Changed
- The refined alignment file is `<TS>.xf` (was `<TS>_refined.xf`).
- `detect_2d` keeps the best 1200 picks per tilt image by default (`--top_k`); `--threshold` selects by score
  instead (it was the default, at -6).
- Residual and selection: the reprojection residual (`rlnJoltResidual`, the FOM's main term) is the RMS over the
  matched tilts, with a new `rlnJoltMatchedTilts` column; the 12 px residual gate is off by default
  (`--residual_threshold` is optional). **Changes the output**: more particles are kept and FOM values change.
- **Changes the default output**: particles are written in the `warp` frame; earlier versions wrote JOLT's own
  frame, which is mirrored in Z relative to Warp tomograms.
- `.mod` files carry the tomogram size in the header and use IMOD's voxel-center convention (index + 0.5).
- `versions.txt` is now `CHANGELOG.md`; the documentation's limitations page was removed.

## [0.0.1_alpha] - 2026-10-07

### Added
- Port of the original JOLT pipeline: `import`, `finetune`, `detect_2d`, `joint_optimize` and `filter`. MLESAC
  initialization, greedy bundle adjustment of particles and per-tilt shifts, the top-5 figure of merit with the
  detector score, residual/border/duplicate selection, refined `.xf` output. Run in the original environment the
  port reproduces the CZI reference runs exactly.
- Package skeleton: `jolt` command, test case `tests/data/czi_TS_5`, documentation, tests and CI.
