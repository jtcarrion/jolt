Conventions
===========

Coordinates and frames
----------------------

* 2D coordinates are in pixels of the tilt stack. Picks are expected in the **raw** (unaligned) stack frame
  by default; the IMOD ``.xf`` is applied to them. Picks made on the aligned stack are registered with
  ``jolt import --picks_frame aligned``.
* Tomogram dimensions (``--tomo_dims_xyz``) are in pixels at the tilt-stack pixel size. X and Y equal the
  aligned tilt-image size; Z is the tomogram thickness.

Tilt order
----------

An IMOD ``.xf`` is positional: row *i* applies to image *i* of the stack. JOLT never sorts or renumbers
tilts; angles come from the ``.tlt`` that belongs to the stack, and a ``.tlt`` whose length disagrees with
the ``.xf`` is an error.

Image center (known issue)
--------------------------

The original code uses two different image-center conventions, ``W/2`` in one place and ``(W-1)/2`` in
another, which differ by half a pixel. The first release reproduces the original behavior exactly; the
convention will be settled (and checked against IMOD and TeamTomo) in a following release.
