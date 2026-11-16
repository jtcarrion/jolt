# JOLT

JOLT finds 3D particle positions in cryo-electron tomography data directly from 2D particle picks on the
tilt images, and jointly refines the tilt-series alignment (per-tilt shifts) together with the particle
positions. It is designed to run on WarpTools projects before tomogram reconstruction.

Workflow:

```
jolt import          # read a Warp project (tilt stacks, .xf, .tlt, .xml) and create jolt.yaml
jolt finetune        # (optional) fine-tune Topaz on known particles
jolt detect_2d       # 2D particle picking on every tilt image with Topaz
jolt joint_optimize  # 3D particle positions + refined alignment, with a per-particle figure of merit
jolt filter          # select particles by figure of merit, residual, tilt series, ...
```

## Installation

```bash
git clone https://github.com/jtcarrion/jolt.git && cd jolt
CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yml -p <prefix>
conda activate <prefix>
pip install --no-deps -e .
jolt --help
```

Topaz (for `finetune` and `detect_2d`) runs from its own environment; point JOLT at it with `--topaz_bin`
or `JOLT_TOPAZ_BIN`.

## Fine-tuned Topaz Weights for In Situ Datasets

A fine-tuned Topaz model for picking in raw tilt images **at 10 Å/pixel** is available on Zenodo: [zenodo.org/records/24144234](https://zenodo.org/records/24144234) (DOI: 10.5281/zenodo.24144234). It was trained with balanced labels on three in situ datasets (EMPIAR-10499 *M. pneumoniae*, EMPIAR-11983 *E. coli*, and CZI DS-10301 *Chlamydomonas*).

Using these weights (or other fine-tuned models) inside JOLT can be easily done with the `jolt detect_2d [...] --weights` flag.

```bash
jolt detect_2d <project> --weights ~/jolt_models/topaz_tilt_generalist_v2_ep20.sav \
    --radius 8 --top_k 1200
jolt joint_optimize <project>
```


**Status:** pre-release (0.0.x). The commands are implemented and validated against the original research
code on the CZI DS-10441 benchmark; interfaces may still change.

## License

JOLT is free software released under the GNU General Public License v3.0; see [LICENSE](LICENSE).
