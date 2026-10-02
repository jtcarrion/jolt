"""Fine-tune the Topaz 2D particle picker on known particles (optional step).

``jolt finetune`` takes known particle positions, either 3D (projected into every raw tilt image through
that tilt series' alignment to make 2D labels) or 2D labels directly, trains Topaz on them, and records the
trained weights in jolt.yaml so that ``jolt detect_2d`` uses them.
"""

import argparse


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt finetune`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("project_dir", help="JOLT project directory (created by jolt import)")

    labels = parser.add_argument_group("Labels (exactly one)")
    source = labels.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels_3d",
        default=None,
        help="STAR file of known 3D particle positions (Angstrom, e.g. a WarpTools/M species STAR); "
        "projected into every raw tilt image to make 2D labels",
    )
    source.add_argument(
        "--labels_2d",
        default=None,
        help="STAR file of 2D labels per tilt image (pixels, raw-stack frame), used as they are",
    )

    training = parser.add_argument_group("Training")
    training.add_argument("--heldout_ts", required=True, help="comma-separated tilt series held out for validation")
    training.add_argument("--exclude_ts", default=None, help="comma-separated tilt series left out entirely")
    training.add_argument(
        "--max_abs_angle",
        type=float,
        default=40.0,
        help="use only tilt images whose absolute tilt angle is <= this (degrees)",
    )
    training.add_argument("--model", default="resnet16", help="Topaz model to start from")
    training.add_argument("--radius", type=int, default=3, help="Topaz training label radius (pixels)")
    training.add_argument(
        "--num_particles", type=float, default=350, help="expected number of particles per tilt image (all of them)"
    )
    training.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    training.add_argument("--minibatch_size", type=int, default=256, help="minibatch size")
    training.add_argument("--epoch_size", type=int, default=100, help="minibatches per epoch")
    training.add_argument("--epochs", type=int, default=20, help="number of epochs; the final epoch's model is used")
    training.add_argument("--seed", type=int, default=0, help="random seed for Topaz (numpy, torch, random)")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--devices", default=None, help="GPU indices to use, e.g. 0,1 (default: all visible)")
    execution.add_argument(
        "--topaz_bin", default=None, help="Topaz executable (default: $JOLT_TOPAZ_BIN, then topaz on PATH)"
    )
    execution.add_argument("--overwrite", action="store_true", help="retrain even if a trained model exists")
    execution.add_argument("--dry_run", action="store_true", help="build and check the training data only")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt finetune``."""
    raise NotImplementedError("`jolt finetune` is not ported yet.")


if __name__ == "__main__":
    main(add_args().parse_args())
