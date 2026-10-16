Limitations
===========

* ``jolt import`` reads WarpTools projects only.
* Tilt series must be aligned already (an IMOD ``.xf`` per tilt series). JOLT refines per-tilt shifts, not
  the tilt axis or the tilt angles.
* Results can differ very slightly between GPU types and library versions (PyTorch, NumPy, SciPy), since bundle
  adjustment is an iterative optimization. The default ``evidence`` loss is smooth (no match switching), so such
  tiny differences do not change the solution: perturbing the starting points by 0.001 px moves refined shifts
  by less than 0.04 px and leaves the selected particles identical on the CZI benchmark. With
  ``--ba_loss nearest`` (the previous behavior) the line search and convergence test keep the shifts within
  about 0.3 px, but individual particles can still move by several pixels.
* The ``evidence`` loss needs the pick localization scale ``--ba_sigma``. By default it is estimated per tilt
  series from the MLESAC hypotheses (2.2-2.4 px on CZI DS-10441, 2.9 px on EMPIAR-10499, 3.1-3.3 px on
  EMPIAR-13391, all at 10 A/px), bounded to 1.5-6 px with a fallback to 2.5 px that is logged as a WARNING and
  recorded in ``run_status.json`` (``ba.noise_model``). The estimate runs about 10% below the value measured
  against ground truth because the hypotheses are the best-fitting points; the results are insensitive to
  that. A value that is too small loses matches at tight scoring radii, too large a value blurs neighboring
  particles. The detection probability ``--ba_gamma`` stays at 0.7 by default: its per-series estimate is
  biased high for the same reason.
* The pre-BA "gap filling" step of the original pipeline is not part of JOLT: with MLESAC initialization and
  greedy bundle adjustment (which re-matches every particle to the picks itself) it had no effect on the
  output.
* ``jolt finetune`` projects 3D labels with the project's geometry as recorded in ``jolt.yaml``; it does not
  fit an unknown volume center, tilt sign or y flip. Give ``--labels_frame`` the frame the labels are in.
* The ``warp`` output frame's sub-voxel offsets and scale come from the Warp XML (``ImageDimensionsAngstrom``,
  ``DimensionsAngstrom``). They were measured on one dataset with a small in-plane rotation (EMPIAR-10499,
  4.3 degrees); for large rotations (e.g. a 90-degree canvas swap) the scale is rotated into volume axes as Warp's
  code implies, which has not been measured yet.
* A single pass refines a misaligned tilt series only part of the way: on EMPIAR-10499 one pass from picks alone
  stops at about 1.1-2.2 px bead RMS and loses recall (initialization runs on the misaligned picks). Use
  ``--iter 3`` for poorly aligned data: each pass re-initializes on the improved alignment (8 px injected errors:
  bead RMS 2.2 -> 1.1 px, recall 50 -> 67 %). Keep ``--iter 1`` for well-aligned series: extra passes drift slightly
  (fiducial alignment: bead RMS 0.84 -> 0.90 px over 3 passes, recall unchanged), and each pass costs one run time.
* With gold beads, the bundle-adjustment health check can flag ``shift_runaway`` when the beads legitimately correct
  a few badly aligned tilts by several pixels (seen on patch-tracking alignments of EMPIAR-10499); check the bead RMS
  in ``run_status.json`` (``fiducials``) before discarding such a series.
* The ``teamtomo`` frame shows an unexplained ``+0.15 +/- 0.05`` voxel z residual against a TeamTomo reconstruction
  (one series). CZI ground truth sits 0.5 voxel below the ``teamtomo`` frame in z (see Conventions).
