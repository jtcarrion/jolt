Getting started
===============

JOLT finds 3D particle positions in cryo-electron tomography directly from 2D particle picks on the tilt
images, and jointly refines the tilt-series alignment (per-tilt shifts) while doing so. It reads WarpTools
projects and writes particles and a refined alignment that go straight back into Warp.

Workflow
--------

.. code-block:: bash

   jolt import --warp_dir <warp_project> --project_dir <project> --tomo_dims_xyz X,Y,Z
   jolt finetune <project> --labels_3d particles.star --heldout_ts TS_1   # optional
   jolt detect_2d <project> --radius 8              # best 1200 picks per tilt image (--top_k), or --threshold
   jolt joint_optimize <project>                    # 3D particles + refined alignment
   jolt filter <project>/joint_optimize -o selected.star --min_fom 0.4

Each command after ``import`` reads the project's ``jolt.yaml`` and records the parameters it used there.
Picks you already have can be registered with ``jolt import --picks_2d`` instead of running ``detect_2d``.
To use the refined alignment and the particles in Warp (reconstruction, particle export), see
:doc:`warp`. The output files are described in :doc:`conventions`.

Try it on the bundled test case
-------------------------------

The repository includes one tilt series of the CZI DS-10441 synthetic benchmark with its 2D picks
(``tests/data/czi_TS_5``; the stack holds blank pixels, so it can be used with existing picks but not with
``detect_2d``). From the repository root, on a machine with a GPU:

.. code-block:: bash

   jolt import --warp_dir tests/data/czi_TS_5 --project_dir ts5_demo --tomo_dims_xyz 630,630,200 \
       --picks_2d "warp_tiltseries/tiltstack/{ts}/{ts}_topaz_picks.star"
   jolt joint_optimize ts5_demo
   jolt filter ts5_demo/joint_optimize -o ts5_demo/selected.star --min_fom 0.4

``ts5_demo/joint_optimize/TS_5/`` then holds about 1,200 particles (``TS_5_particles.star``, also as
``TS_5_particles.mod`` for ``3dmod``), the refined alignment (``TS_5.xf`` and ``TS_5.tlt``) and
``run_status.json``. A run takes a few minutes on one GPU.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   warp
   commands/index
   conventions
