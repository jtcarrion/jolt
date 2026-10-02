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

Image center
------------

All image and tomogram centers are ``((N - 1)/2)`` in 0-based pixel coordinates, where an integer coordinate
is the center of a pixel (the Topaz and STAR convention). This is what IMOD's ``newstack -xf`` uses: applied to
images with Gaussian blobs at known positions, its translations match ``(N - 1)/2`` to 0.003 px, whereas
``N/2`` is off by up to about 0.45 px when the alignment rotates the image or changes the canvas. Versions
before 0.0.1 used ``N/2`` in the raw-to-aligned pick transform, which displaced particles by up to about 1 px
on rotated, reshaped canvases (bmp6-like geometry) without any visible change in the residuals.

Figure of merit
---------------

``rlnAutopickFigureOfMerit`` is ``0.96 geom + 0.01 (match_fraction + global_coverage + avg_dist_quality +
rmse_quality)``, with ``geom = exp(-residual / 6)``. ``rlnJoltDetectorScore`` is the mean 2D detector score of
the picks matched to the particle (in the 2D picker's own units) and ``rlnJoltResidual`` its reprojection
residual in pixels. Two details are kept from the original code for identical output in 0.0.1: the residual
is measured with the input alignment rather than the refined shifts, and the four small benchmark terms
compare projections that are not mirrored in y.
