JOLT
====

JOLT finds 3D particle positions in cryo-electron tomography directly from 2D particle picks on the tilt
images, and jointly refines the tilt-series alignment while doing so.

Workflow
--------

.. code-block:: bash

   jolt import --warp_dir <warp_project> --project_dir <project> --tomo_dims_xyz X,Y,Z
   jolt finetune <project> --labels_3d particles.star --heldout_ts TS_1   # optional
   jolt detect_2d <project> --radius 8
   jolt joint_optimize <project>                       # --output_frame warp (default) or teamtomo
   jolt filter <project>/joint_optimize -o selected.star --min_fom 0.3

Each command after ``import`` reads the project's ``jolt.yaml`` and records the parameters it used there.
Picks you already have can be registered with ``jolt import --picks_2d`` instead of running ``detect_2d``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   commands/index
   conventions
   limitations
