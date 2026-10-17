Conventions
===========

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Content
   * - ``detect_2d/<TS>/<TS>_picks.star``
     - 2D picks on the raw tilt images (pixel x, y, Topaz score, tilt image by stack position)
   * - ``detect_2d/<TS>/<TS>_picks.mod``
     - the same picks as an IMOD model for the raw stack: ``3dmod <TS>.st <TS>_picks.mod`` (one contour per tilt
       image, each pick drawn only on its own tilt, circle radius = ``--radius``)
   * - ``joint_optimize/<TS>/<TS>_particles.star``
     - 3D particles with ``rlnAutopickFigureOfMerit``, ``rlnJoltDetectorScore``, ``rlnJoltResidual`` and
       ``rlnJoltMatchedTilts`` (see `Figure of merit`_)
   * - ``joint_optimize/<TS>/<TS>_particles.mod``
     - the same particles as an IMOD model for the tomogram
   * - ``joint_optimize/<TS>/<TS>.xf``, ``<TS>.tlt``
     - the refined alignment and its tilt angles, ready for Warp (see `Refined alignment`_ and :doc:`warp`)
   * - ``joint_optimize/<TS>/run_status.json``
     - what happened: picks, MLESAC counts, bundle adjustment status and noise model, output frame, deselected
       tilts, per-pass results with ``--iter``
   * - ``joint_optimize/summary.json``
     - one line per tilt series; ``joint_optimize`` lists every tilt series that needs attention at the end

2D picks
--------

``jolt detect_2d`` keeps the best 1200 picks on each tilt image by default (``--top_k``). ``--threshold T`` keeps
every pick with a Topaz score above ``T`` instead (``-6`` keeps nearly everything with the pretrained model).
``--radius`` is Topaz's extraction radius: two picks are never closer than this, so set it to about the radius
of the smallest particle you want. JOLT discards inconsistent picks in 3D, but cannot recover a particle that was
not picked on enough tilts, so keep the 2D selection generous.

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
the ``.xf`` or the stack stops ``jolt import``.

Tilt images deselected in Warp (``UseTilt`` false in the XML) are ignored: their picks take no part in the
reconstruction, and ``run_status.json`` lists them. A stack written with only the used tilts (the XML then lists
more tilts than the stack) is used as is.

Refined alignment
-----------------

``joint_optimize`` writes ``<TS>.xf`` and ``<TS>.tlt`` in ``<project>/joint_optimize/<TS>/``: every input row with
its translation changed by the shift that bundle adjustment found, in the correction convention IMOD's
``newstack -xf`` applies, and the stack's own ``.tlt`` lines copied verbatim (JOLT never changes tilt angles). Rows
keep the input order. Tilt images deselected in Warp (``UseTilt``) are left out of both files, because Warp maps
the rows onto its in-use tilts (``run_status.json`` lists the stack positions written). If bundle adjustment fails,
neither file is written (``run_status.json`` says so).

Bundle adjustment
-----------------

``joint_optimize`` initializes particles with MLESAC and then refines the particles and the per-tilt shifts
together. The default loss (``--ba_loss evidence``) scores each particle in each tilt by the evidence of all
picks within the match distance, using a pick localization scale ``sigma`` estimated per tilt series
(``--ba_sigma auto``, bounded to 1.5-6 px with a fallback to 2.5 px; the value used is in ``run_status.json``
under ``ba.noise_model`` and in ``summary.json``). ``--ba_loss nearest`` restores the previous loss (nearest pick
only).

``--iter N`` runs N passes, each re-initializing on the previous pass's refined alignment; it helps on poorly
aligned tilt series (``--iter 3``). It needs picks on the raw tilt images (the default): picks registered with
``--picks_frame aligned`` do not move when the alignment is refined, so use ``--iter 1`` with them.

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
* The refined ``.xf`` does not depend on the output frame. The output folder imports into Warp as-is
  (``ts_import_alignments --alignments <project>/joint_optimize --alignment_angpix <pixel size>``, then ``ts_ctf``
  before ``ts_reconstruct``; rows follow the stack order, which must be Warp's tilt-stack order): Warp reads
  ``<TS>/<TS>.xf`` and ``<TS>/<TS>.tlt``, keeps IMOD's matrix and converts the shifts about its own image center,
  exactly as it did for the original etomo ``.xf``, and takes the tilt angles from the ``.tlt``.

Gold beads (fiducials)
----------------------

``jolt import --fiducials imod`` registers each tilt series' etomo bead model (``<ts>.fid``, tracked on the
coarse-prealigned stack) together with its ``<ts>.prexg``. Every observation is mapped back to the raw frame (inverse
``.prexg`` about ``(N - 1)/2``, after removing IMOD's +0.5 model convention) and written to
``<project>/fiducials/<ts>_fiducials_2d.star`` with a ``rlnJoltBeadId`` column. ``joint_optimize`` maps them to the
aligned frame with the ``.xf``, like the picks, and adds them to bundle adjustment with their **known tracks**:
each bead is one 3D point refined together with the per-tilt shifts, and ``--fiducial_weight`` (default 1.0) times
the mean Huber loss over bead observations is added to the pick term. Beads are never written to the particle STAR;
they go to ``<ts>_fiducials.star`` (same output frame, with each bead's RMS residual), and ``run_status.json``
records the bead RMS under the input and the refined alignment.

On EMPIAR-10499 (00254–00258, half the beads anchoring and the other half held out for scoring), the refined
alignment from any start (injected per-tilt errors of ±3/5/8 px, or a patch-tracking alignment at 2.3 px) reaches
0.84–0.85 px held-out bead RMS, the same as a fit to the beads alone, where picks alone stop at 1.1–2.2 px. Recall
of the particles also rises (by up to 6 points at 50 A). Removing Topaz picks near beads and a FOM-weighted second
pass were tested and did not help.

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
the picks matched to the particle (in the 2D picker's own units), ``rlnJoltResidual`` the reprojection residual in
pixels and ``rlnJoltMatchedTilts`` the number of tilts it is computed over.

The residual is the RMS distance from the particle's projections to the nearest pick over the **matched tilts**:
the tilts with a pick within the match distance (``--match_distance``, 12 px), which are the tilts bundle
adjustment fits. Tilts where the 2D picker found nothing near the particle do not count against it. Versions before
0.0.1 used the RMS over all tilts and dropped particles above 12 px; on EMPIAR-10499 (four pick sets) and both CZI
cohorts that gate removed true particles without improving precision (a picker trained only on low tilts lost 12
points of recall at 50 A), and the matched residual ranks true above false particles better on every set (AUROC
+0.02 to +0.22). ``joint_optimize`` therefore applies no residual threshold unless ``--residual_threshold`` is set;
rank and filter with the figure of merit (``jolt filter --min_fom``) instead.

Choosing a FOM cut. FOM values cover a similar range on every dataset measured (10th-90th percentile about
0.37-0.66 on CZI DS-10441 and EMPIAR-10499), so fixed cuts transfer:

* ``--min_fom 0.4``: a near-lossless clean-up. It removed 3-24 % of the particles for at most 1 point of recall with
  good picks (EMPIAR-10499, fine-tuned Topaz: recall at 50 A 74.7 -> 73.8 %, precision 23 -> 30 %; CZI F-beta1 rises
  on both cohorts, F-beta4 changes by at most 0.012). A good default.
* ``--min_fom 0.5``: about twice the precision for a recall cost (EMPIAR-10499 fine-tuned: precision 23 -> 54 %,
  recall 75 -> 51 %; CZI fine-tuned precision 0.48 -> 0.64). For subtomogram averaging, where purity matters more
  than count.
* Above 0.55 few particles remain. Cutting does not rescue a weak 2D picker: with zero-shot Topaz picks on
  EMPIAR-10499 no cut raises precision above 10 %.

The residual (and so the selection), the FOM, its benchmark terms and the detector score project each
particle with the alignment written to the refined ``.xf`` (the input alignment plus the shifts bundle
adjustment changed), and compare those projections with the picks in the picks' own frame. Versions before
0.0.1 used the input alignment, so on a misaligned tilt series good particles kept large residuals and low FOM
values after bundle adjustment had corrected the misalignment.
