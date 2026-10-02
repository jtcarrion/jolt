Limitations
===========

* Only WarpTools projects are read by ``jolt import``.
* Tilt series must be aligned already (an IMOD ``.xf`` per tilt series); JOLT refines per-tilt shifts, not
  the tilt axis or tilt angles.
* Bundle-adjustment results depend slightly on the GPU type.
