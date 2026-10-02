Limitations
===========

* ``jolt import`` reads WarpTools projects only.
* Tilt series must be aligned already (an IMOD ``.xf`` per tilt series). JOLT refines per-tilt shifts, not
  the tilt axis or the tilt angles.
* Results depend slightly on the GPU type and on library versions (PyTorch, NumPy, SciPy): bundle adjustment
  is an iterative optimization, so tiny rounding differences can lead to slightly different solutions. On the
  CZI benchmark the effect is far below the scatter between tilt series.
* The pre-BA "gap filling" step of the original pipeline is not part of JOLT: with MLESAC initialization and
  greedy bundle adjustment (which re-matches every particle to the picks itself) it had no effect on the
  output.
* ``jolt finetune`` projects 3D labels with the project's geometry as recorded in ``jolt.yaml``; it does not
  fit an unknown volume center, tilt sign or y flip.
