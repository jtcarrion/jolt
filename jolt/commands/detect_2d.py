"""Pick 2D particles on every raw tilt image with Topaz and add the picks to jolt.yaml.

``jolt detect_2d`` splits each raw tilt stack into tilt images, runs Topaz (preprocessing and extraction,
seeded for reproducibility), writes one 2D pick STAR file per tilt series, and records the pick files in
jolt.yaml for ``jolt joint_optimize``. Skip it if you registered your own picks with
``jolt import --picks_2d``.
"""

import argparse


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the ``jolt detect_2d`` arguments to ``parser`` (or to a new parser) and return it."""
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("project_dir", help="JOLT project directory (created by jolt import)")

    picking = parser.add_argument_group("Picking")
    picking.add_argument(
        "--radius",
        type=int,
        required=True,
        help="Topaz extraction radius in pixels of the tilt stack (about the particle radius); "
        "two picks are never closer than this",
    )
    selection = picking.add_mutually_exclusive_group()
    selection.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="keep picks with a Topaz score above this (default: -6 when --top_k is not given)",
    )
    selection.add_argument("--top_k", type=int, default=None, help="keep the best K picks on each tilt image")
    picking.add_argument(
        "--weights",
        default=None,
        help="Topaz model: a pretrained name (e.g. resnet16) or a .sav file "
        "(default: the jolt finetune weights in jolt.yaml if present, else resnet16)",
    )
    picking.add_argument("--seed", type=int, default=0, help="seed for Topaz preprocessing (same seed, same picks)")

    execution = parser.add_argument_group("Execution")
    execution.add_argument("--ts", default=None, help="comma-separated tilt-series names to process (default: all)")
    execution.add_argument("--devices", default=None, help="GPU indices to use, e.g. 0,1, or 'cpu' (default: all)")
    execution.add_argument(
        "--topaz_bin", default=None, help="Topaz executable (default: $JOLT_TOPAZ_BIN, then topaz on PATH)"
    )
    execution.add_argument("--overwrite", action="store_true", help="redo tilt series whose picks already exist")
    execution.add_argument(
        "--no_save_tilts", action="store_true", help="do not keep the preprocessed tilt images Topaz picked on"
    )
    execution.add_argument("--work_dir", default=None, help="scratch directory (default: a folder under $TMPDIR)")
    execution.add_argument("--keep_work", action="store_true", help="keep each tilt series' scratch folder")
    execution.add_argument("--dry_run", action="store_true", help="list what would be processed, then exit")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run ``jolt detect_2d``."""
    raise NotImplementedError("`jolt detect_2d` is not ported yet.")


if __name__ == "__main__":
    main(add_args().parse_args())
