Using JOLT with WarpTools
=========================

JOLT reads a WarpTools project (``jolt import --warp_dir``) whose tilt series are already aligned (an IMOD
``.xf`` and ``.tlt`` per tilt series, e.g. from ``ts_etomo_fiducials``, ``ts_etomo_patches`` or
``ts_import_alignments``). It refines the per-tilt shifts, not the tilt axis or the tilt angles, and writes
everything Warp needs to use that refined alignment and the particles.

1. Import the refined alignment
-------------------------------

``joint_optimize`` writes ``<TS>/<TS>.xf`` and ``<TS>/<TS>.tlt`` for every tilt series in
``<project>/joint_optimize/``, which is the layout ``ts_import_alignments`` expects, so the folder is passed as
is. Import into a copy of the processing folder to keep the original alignment:

.. code-block:: bash

   mkdir warp_jolt && cp warp_tiltseries/*.xml warp_jolt/
   WarpTools ts_import_alignments --settings warp_tiltseries.settings \
       --input_processing warp_jolt --output_processing warp_jolt \
       --alignments <project>/joint_optimize --alignment_angpix <tilt-stack pixel size>

* ``--alignment_angpix`` is the pixel size of the tilt stacks JOLT used (recorded in ``jolt.yaml``).
* Warp takes its tilt angles from ``<TS>.tlt``, which JOLT copies unchanged from the stack's own ``.tlt``.
* Tilts deselected in Warp (``UseTilt``) are left out of both files, so Warp keeps them deselected. JOLT reads the
  current deselection from each Warp XML when ``joint_optimize`` runs.
* A tilt series without a refined alignment (bundle adjustment failed, or not processed) has no folder; Warp lists
  it in ``failed_items.json`` and keeps its previous alignment.

2. CTF and reconstruction
-------------------------

The defocus fit depends on the tilt geometry, so re-run ``ts_ctf`` before reconstructing:

.. code-block:: bash

   WarpTools ts_ctf --settings warp_tiltseries.settings --input_processing warp_jolt --output_processing warp_jolt ...
   WarpTools ts_reconstruct --settings warp_tiltseries.settings --input_processing warp_jolt --output_processing warp_jolt ...

3. Export the particles
-----------------------

Combine the particles of all tilt series into one STAR file (optionally with a figure-of-merit cut, see
:doc:`conventions`) and export them from the processing folder that holds the JOLT alignment:

.. code-block:: bash

   jolt filter <project>/joint_optimize -o particles.star --min_fom 0.4
   WarpTools ts_export_particles --settings warp_tiltseries.settings --input_processing warp_jolt \
       --input_star particles.star --coords_angpix <tilt-stack pixel size> \
       --output_star relion/particles.star --output_angpix <A/px> --box <px> --diameter <A> --3d   # or --2d

* Coordinates are tilt-stack pixels in the frame of Warp's tomograms (``joint_optimize --output_frame warp``, the
  default), so ``--coords_angpix`` is the tilt-stack pixel size.
* Export from the folder where you imported JOLT's alignment. Particles and tomograms from different alignments do
  not line up: a refined alignment can move the whole tomogram by a voxel or two.
* ``ts_export_particles --input_directory`` does not search sub-folders; use the combined STAR from ``jolt filter``.
