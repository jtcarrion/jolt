"""jolt finetune training settings: class prior, Topaz arguments, checkpoint names and best-epoch selection."""

import numpy as np
import pytest

from jolt import topaz
from jolt.commands.finetune import add_args, class_prior_settings


def _topaz_fixed_disc(radius):
    """Topaz's pixels_given_radius for 2D as it should be (and is, once fixed): a 2D lattice disc."""
    grid = np.linspace(-radius, radius, 2 * radius + 1)
    xx, yy = np.meshgrid(grid, grid)
    return int((xx**2 + yy**2 <= radius**2).sum())


@pytest.mark.parametrize("radius", [0, 1, 2, 3, 4, 6, 8])
def test_disc_pixels_is_the_2d_disc(radius):
    assert topaz.disc_pixels(radius) == _topaz_fixed_disc(radius)


def test_disc_pixels_known_values():
    # 29 at r = 3: Topaz 0.3.18's stock count is 7x larger (203), which is the bug the explicit --pi avoids
    assert topaz.disc_pixels(3) == 29
    assert topaz.disc_pixels(6) == 113


def test_class_prior_formula():
    assert topaz.class_prior(350, 3, 803 * 803) == pytest.approx(350 * 29 / (803 * 803))


@pytest.mark.parametrize("epochs", [1, 5, 9, 10, 12, 20, 99, 100, 120])
def test_checkpoint_names_match_topaz(epochs, tmp_path):
    digits = int(np.ceil(np.log10(epochs)))  # Topaz 0.3.18 training.py
    for epoch in {1, epochs // 2 or 1, epochs}:
        expected = "model" + ("_epoch{:0" + str(digits) + "}.sav").format(epoch)
        assert topaz.checkpoint_path(tmp_path, epoch, epochs).name == expected


def _write_curve(path, test_auprc):
    head = "epoch\titer\tsplit\tloss\tge_penalty\tprecision\tadjusted_precision\ttpr\tfpr\tauprc\n"
    rows = []
    for epoch, auprc in enumerate(test_auprc, start=1):
        rows.append(f"{epoch}\t{epoch * 10 - 5}\ttrain\t0.5\t1.0\t0.2\t0.3\t0.2\t0.05\t-\n")
        rows.append(f"{epoch}\t{epoch * 10}\ttest\t0.06\t-\t0.02\t0.04\t0.17\t0.05\t{auprc}\n")
    path.write_text(head + "".join(rows))


def test_auprc_by_epoch_reads_only_test_rows(tmp_path):
    _write_curve(tmp_path / "train_curve.txt", [0.031, 0.044, 0.040])
    assert topaz.auprc_by_epoch(tmp_path / "train_curve.txt") == {1: 0.031, 2: 0.044, 3: 0.040}
    assert topaz.auprc_by_epoch(tmp_path / "missing.txt") == {}


def test_best_epoch_ties_go_to_the_earlier_epoch():
    assert topaz.best_epoch({1: 0.03, 2: 0.05, 3: 0.05, 4: 0.04}) == 2
    assert topaz.best_epoch({}) is None


def test_class_prior_settings_default_and_explicit():
    s = class_prior_settings(n_images=100, n_labels=17470, mean_image_area=803 * 803, radius=3)
    assert s["labels_per_image"] == pytest.approx(174.7)
    assert s["num_particles"] == pytest.approx(1.5 * 174.7)
    assert s["pi"] == pytest.approx(1.5 * 174.7 * 29 / (803 * 803))
    assert s["pi"] > s["p_observed_estimate"]
    s = class_prior_settings(100, 17470, 803 * 803, 3, num_particles=350)
    assert s["num_particles"] == 350


def test_class_prior_settings_refuses_pn():
    # the old fixed default (350) on a dense set (~360 labels per image) is exactly the silent-PN case
    with pytest.raises(SystemExit, match="must exceed"):
        class_prior_settings(100, 36000, 652 * 630, 3, num_particles=350)
    with pytest.raises(SystemExit):
        class_prior_settings(0, 0, 803 * 803, 3)


def test_train_args_pass_pi_units_and_no_topaz_pi(tmp_path):
    args = topaz.train_args(
        tmp_path, tmp_path / "run", model="resnet16", units=64, radius=3, pi=0.0117, learning_rate=2e-4,
        minibatch_size=256, epoch_size=1000, epochs=12, device=0,
    )  # fmt: skip
    assert args[args.index("-n") + 1] == "0"  # Topaz uses --pi only when -n is not positive
    assert float(args[args.index("--pi") + 1]) == pytest.approx(0.0117)
    assert args[args.index("--units") + 1] == "64"
    assert args[args.index("--method") + 1] == "GE-binomial"
    assert args[args.index("--epoch-size") + 1] == "1000" and args[args.index("--num-epochs") + 1] == "12"


def test_cli_defaults_follow_the_benchmark():
    a = add_args().parse_args(["proj", "--labels_2d", "x.star", "--heldout_ts", "TS_1"])
    assert (a.epochs, a.epoch_size, a.learning_rate, a.units, a.minibatch_size) == (12, 1000, 2e-4, 64, 256)
    assert a.num_particles is None and a.num_particles_factor == 1.5
    assert (a.radius, a.max_abs_angle, a.model) == (3, 40.0, "resnet16")
