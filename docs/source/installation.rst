Installation
============

JOLT needs Linux, Python 3.11 to 3.13, and an NVIDIA GPU (driver supporting CUDA 12.4 or newer) for
practical run times. JOLT runs on the CPU too, just several times slower. The recommended environment uses
conda-forge only, so that numpy and PyTorch share a single OpenMP runtime.

.. code-block:: bash

   git clone https://github.com/jtcarrion/jolt.git
   cd jolt
   CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yml -p <prefix>
   conda activate <prefix>
   pip install --no-deps -e .
   jolt --help
   pytest            # optional: the test suite runs on the CPU in a few minutes

``CONDA_OVERRIDE_CUDA`` is only needed when you build the environment on a machine without a GPU (e.g. a login
node); set it to the CUDA version your GPU nodes' driver supports.

Topaz
-----

``jolt detect_2d`` and ``jolt finetune`` run `Topaz <https://github.com/tbepler/topaz>`_ natively from its own
environment. Other 2D CNN models as well as 2D template matching is supoprted via ``jolt import --picks_2d``.
Build Topaz with a CUDA PyTorch from conda-forge, then install Topaz itself without dependencies:

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.4 conda create -y -p <topaz_prefix> -c conda-forge --override-channels \
       python=3.10 "pytorch=*=cuda*" torchvision numpy scipy pandas scikit-learn pillow future h5py "tqdm>=4.65"
   <topaz_prefix>/bin/pip install --no-deps topaz-em==0.3.18

Check on a GPU node that Topaz's PyTorch can use the GPU:

.. code-block:: bash

   <topaz_prefix>/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True

If it prints ``False``, the environment holds a CPU-only PyTorch: Topaz then prints "Falling back to CPU" and
keeps going, about ten times slower (roughly 10 minutes instead of 1 per tilt series). Point JOLT at the Topaz
executable with ``--topaz_bin <topaz_prefix>/bin/topaz`` or the ``JOLT_TOPAZ_BIN`` environment variable. Picks from
Topaz on the GPU and on the CPU agree to about 99.8 %.
