# Changelog

All notable changes to JOLT are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/).
Entries marked **changes the output** alter the particles or the alignment JOLT writes for the same input.

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
