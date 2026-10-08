Limitations
===========

* ``jolt import`` reads WarpTools projects only.
* Tilt series must be aligned already (an IMOD ``.xf`` per tilt series). JOLT refines per-tilt shifts, not
  the tilt axis or the tilt angles.
* Results can differ very slightly between GPU types and library versions (PyTorch, NumPy, SciPy), since bundle
  adjustment is an iterative optimization. Bundle adjustment uses a line search and a convergence test, so
  such tiny differences no longer change the solution: perturbing the starting points by 0.001 px moves refined
  shifts by at most about 0.3 px on the CZI benchmark.
* The pre-BA "gap filling" step of the original pipeline is not part of JOLT: with MLESAC initialization and
  greedy bundle adjustment (which re-matches every particle to the picks itself) it had no effect on the
  output.
* ``jolt finetune`` projects 3D labels with the project's geometry as recorded in ``jolt.yaml``; it does not
  fit an unknown volume center, tilt sign or y flip. Give ``--labels_frame`` the frame the labels are in.
* The ``warp`` output frame's sub-voxel offsets and scale come from the Warp XML (``ImageDimensionsAngstrom``,
  ``DimensionsAngstrom``). They were measured on one dataset with a small in-plane rotation (EMPIAR-10499,
  4.3 degrees); for large rotations (e.g. a 90-degree canvas swap) the scale is rotated into volume axes as Warp's
  code implies, which has not been measured yet.
* Without gold beads, the alignment refined from picks alone converges to about 1.3-2.2 px bead RMS on
  EMPIAR-10499 (limited by per-particle pick scatter): alignments worse than that improve, but picks alone cannot
  reach fiducial quality. Register beads with ``jolt import --fiducials imod`` when an etomo ``.fid`` exists.
* With gold beads, the bundle-adjustment health check can flag ``shift_runaway`` when the beads legitimately correct
  a few badly aligned tilts by several pixels (seen on patch-tracking alignments of EMPIAR-10499); check the bead RMS
  in ``run_status.json`` (``fiducials``) before discarding such a series.
* The ``teamtomo`` frame shows an unexplained ``+0.15 +/- 0.05`` voxel z residual against a TeamTomo reconstruction
  (one series). CZI ground truth sits 0.5 voxel below the ``teamtomo`` frame in z (see Conventions).
