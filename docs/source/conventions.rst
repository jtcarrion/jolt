Conventions
===========

Units and frames
----------------

* All coordinates are in pixels of the tilt stack (the pixel size is read from the ``.st`` header).
* **2D picks** are expected on the **raw** (unaligned) tilt images by default; each tilt's IMOD ``.xf`` is
  applied to them. Picks made on the aligned stack are registered with ``jolt import --picks_frame aligned``.
* Tilt images are identified by their **1-based stack index** in ``rlnMicrographName`` (``tilt_07``,
  ``TS_001_07``); the tilt angle is the ``.tlt`` entry at that index.
* **Tomogram dimensions** (``--tomo_dims_xyz``) are in pixels at the tilt-stack pixel size. X and Y equal the
  aligned tilt-image size; Z is the thickness, which sets the 3D center.
* **3D particles** in the output STAR files are corner-frame tomogram pixels with y pointing down (the
  RELION/IMOD convention), the same frame the CZI ground truth uses. The ``.mod`` file holds the same
  coordinates.

Tilt order
----------

An IMOD ``.xf`` is positional: row *i* applies to image *i* of the stack. JOLT never sorts or renumbers
tilts; angles come from the ``.tlt`` that belongs to the stack, and a ``.tlt`` whose length disagrees with
the ``.xf`` (or the stack, or the Warp ``UseTilt`` count) stops ``jolt import``.

Refined alignment
-----------------

``joint_optimize`` writes ``<TS>_refined.xf``: every input row with its translation changed by the shift that
bundle adjustment found, in the correction convention IMOD's ``newstack -xf`` applies. Rows keep the input
order. If bundle adjustment fails, no refined ``.xf`` is written (``run_status.json`` says so).

Image center (known issue)
--------------------------

The raw-to-aligned pick transform rotates about ``(W/2, H/2)``, while the rest of the geometry uses
``((W - 1)/2, (H - 1)/2)``; the two differ by half a pixel. Version 0.0.1 keeps both exactly as the original
code had them so that its output is unchanged; the convention will be settled (and checked against IMOD and
TeamTomo) in a following release.

Figure of merit
---------------

``rlnAutopickFigureOfMerit`` is ``0.96 geom + 0.01 (match_fraction + global_coverage + avg_dist_quality +
rmse_quality)``, with ``geom = exp(-residual / 6)``. ``rlnJoltDetectorScore`` is the mean 2D detector score of
the picks matched to the particle (in the 2D picker's own units) and ``rlnJoltResidual`` its reprojection
residual in pixels. Two details are kept from the original code for identical output in 0.0.1: the residual
is measured with the input alignment rather than the refined shifts, and the four small benchmark terms
compare projections that are not mirrored in y.
