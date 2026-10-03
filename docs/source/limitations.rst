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
  fit an unknown volume center, tilt sign or y flip.
