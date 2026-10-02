JOLT
====

JOLT finds 3D particle positions in cryo-electron tomography directly from 2D particle picks on the tilt
images, and jointly refines the tilt-series alignment while doing so.

.. warning::

   JOLT is being ported from research code. The commands below are documented, but most are not
   implemented yet.

Workflow
--------

.. code-block:: bash

   jolt import --warp_dir <warp_project> --project_dir <project> --tomo_dims_xyz X,Y,Z
   jolt finetune <project> --labels_3d particles.star --heldout_ts TS_1   # optional
   jolt detect_2d <project> --radius 8
   jolt joint_optimize <project>
   jolt filter <project>/<output> -o selected.star --min_fom 0.3

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   commands/index
   conventions
   limitations
