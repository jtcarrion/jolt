# JOLT

JOLT finds 3D particle positions in cryo-electron tomography data directly from 2D particle picks on the
tilt images, and jointly refines the tilt-series alignment (per-tilt shifts) together with the particle
positions. It is designed to run on WarpTools projects before tomogram reconstruction.

Planned workflow:

```
jolt import          # read a Warp project (tilt stacks, .xf, .tlt, .xml) and create jolt.yaml
jolt finetune        # (optional) fine-tune Topaz on known particles
jolt detect_2d       # 2D particle picking on every tilt image with Topaz
jolt joint_optimize  # 3D particle positions + refined alignment, with a per-particle figure of merit
jolt filter          # select particles by figure of merit, residual, tilt series, ...
```

**Status:** under active development; not yet ready for use.

## License

JOLT is free software released under the GNU General Public License v3.0; see [LICENSE](LICENSE).
