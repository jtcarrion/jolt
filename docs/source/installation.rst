Installation
============

JOLT needs Linux, Python 3.11 to 3.13, and an NVIDIA GPU (driver supporting CUDA 12.4 or newer) for
practical run times. The recommended environment uses conda-forge only, so that numpy and PyTorch share
a single OpenMP runtime.

.. code-block:: bash

   git clone https://github.com/jtcarrion/jolt.git
   cd jolt
   CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yml -p <prefix>
   conda activate <prefix>
   pip install --no-deps -e .
   jolt --help

Topaz (for ``jolt finetune`` and ``jolt detect_2d``) is installed separately; point JOLT at it with
``--topaz_bin`` or the ``JOLT_TOPAZ_BIN`` environment variable.
