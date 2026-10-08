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
* **3D particles** in the output STAR files are 0-based, corner-origin tomogram voxels at the tilt-stack pixel
  size, with y pointing down, written in the frame chosen with ``--output_frame`` (``warp`` by default; see
  `Output frame`_). The ``.mod`` file holds the same particles in IMOD's model convention (voxel centers at
  index + 0.5, the volume size in the header).

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

Output frame
------------

Tomograms of the same tilt series reconstructed by different packages do not share one 3D frame. JOLT computes
in its own frame and writes particle coordinates in the frame of the tomograms you work with
(``joint_optimize --output_frame``; ``finetune --labels_frame`` reads 3D labels in the same way):

``warp`` (default)
    The frame of WarpTools ``ts_reconstruct`` tomograms, which is what ``ts_export_particles`` reads. Warp projects
    a volume point with ``u = cos(theta) x - sin(theta) z`` while IMOD, TeamTomo and JOLT use ``+ sin(theta) z``,
    so Warp tomograms are mirrored in Z (``WarpLib/TiltSeries/TiltSeries.cs``, ``GetPositionInAllTilts``). Warp
    also centers the volume at ``DimensionsAngstrom / 2`` and maps its physical image frame onto the binned
    images with ``SizeRoundingFactors``. With ``D`` the XML ``DimensionsAngstrom``, ``p`` the pixel size and
    ``s`` Warp's size-rounding factors (binned size / (``ImageDimensionsAngstrom`` / ``p``)), each particle at
    ``(x, y, z)`` in JOLT's frame is written as

    ``X = D_x/2p + (x - N_x/2) / s_x``,  ``Y = D_y/2p + (y - N_y/2) / s_y``,
    ``Z = D_z/2p - (z - (N_z - 1)/2) / s_x``

    (the two image scales are rotated into volume axes by the ``.xf`` in-plane rotation). Both dimensions are read
    from each tilt series' Warp XML. If the XML does not have them, exact binning is assumed (``X = x``, ``Y = y``,
    ``Z = N_z - 0.5 - z``) and a warning is logged. Measured on EMPIAR-10499 (5 tilt series, 50 IMOD-tracked gold
    beads, 652 x 630 x 340 voxels at 10 A, where the transform is ``(x + 0.33, y + 0.44, 339.55 - z)`` at the
    center and 0.1 % larger toward the edges): written positions agree with the beads in the Warp tomograms to a
    mean of 0.05 voxel per axis (rms 0.28 voxel). Use the STAR directly with ``WarpTools ts_export_particles
    --input_star <TS>_particles.star --coords_angpix <pixel size>`` (it reads ``rlnCoordinateX/Y/Z`` as
    corner-origin positions and ignores ``rlnCenteredCoordinate*``).
``teamtomo``
    The frame of TeamTomo ``torch-reconstruct-tomogram`` (``teamtomo/teamtomo``), which has JOLT's hand and also
    matches IMOD reconstructions rotated with ``clip rotx`` / ``trimvol -rx`` and the CZI portal tomograms.
    TeamTomo centers the volume at ``N//2``, so the frame is JOLT's with ``z + 0.5`` when ``N_z`` is even
    (``z`` when odd); x and y are unchanged because TeamTomo's image and volume centers cancel. Measured with
    synthetic data through TeamTomo's own etomo loader (``+0.42`` voxel by a slightly biased estimator) and on the
    EMPIAR-10499 beads (``+0.51`` to ``+0.54``). Particles written in this frame and measured in a TeamTomo
    reconstruction of the same tilt series show a residual of ``+0.15 +/- 0.05`` voxel in z (17 beads, one
    series), not yet explained.

Notes:

* **CZI ground truth** (DS-10441) is in JOLT's own frame, i.e. the ``teamtomo`` frame minus 0.5 voxel in z: its
  projections reproduce the CZI 2D ground truth to 0.0004 px through JOLT's geometry. For exact work with CZI
  labels, add 0.5 to their z before ``finetune --labels_frame teamtomo``. The CZI benchmark uses
  ``--output_frame teamtomo``; the 0.5-voxel (5 A) offset is far below the scoring radii.
* Every run records its frame (centers, matrix, z factor and where the values came from) in
  ``run_status.json``, and the frame name in ``summary.json``.
* The refined ``.xf`` does not depend on the output frame. It imports into Warp as-is
  (``ts_import_alignments --alignment_angpix <pixel size>``; rows follow the stack order, which must be Warp's
  tilt-stack order): Warp keeps IMOD's matrix and converts the shifts about its own image center, exactly as it
  did for the original etomo ``.xf``.

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
the picks matched to the particle (in the 2D picker's own units) and ``rlnJoltResidual`` the reprojection
residual in pixels used by the selection step (``--residual_threshold``).

The residual (and so the selection), the FOM, its benchmark terms and the detector score project each
particle with the alignment written to the refined ``.xf`` (the input alignment plus the shifts bundle
adjustment changed), and compare those projections with the picks in the picks' own frame. Versions before
0.0.1 used the input alignment, so on a misaligned tilt series good particles kept large residuals and low FOM
values after bundle adjustment had corrected the misalignment.
