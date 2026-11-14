"""Run inside Topaz's own Python interpreter (never imported by JOLT): ``topaz train`` and checkpoint averaging.

JOLT runs Topaz as an external program and does not import Topaz or its PyTorch. ``jolt finetune`` executes this
file with the interpreter of the Topaz environment:

``python _topaz_train.py train <seed> <topaz train arguments>``
    Seed ``random``, ``numpy`` and ``torch``, replace Topaz 0.3.18's positive-crop sampler with an O(1) version, then
    run ``topaz train``. Stock Topaz draws each positive crop with ``DataFrame.sample()``, which costs time
    proportional to the number of labeled pixels (0.5 s per step at ~41k labels, ~6 s at ~380k). The replacement
    draws one row uniformly at random from the same table (``iloc`` at a random index), so the training crops
    follow the same distribution. The patch is applied only if Topaz is 0.3.18 and the stock method's source has
    the expected SHA-256; otherwise stock Topaz runs unchanged. Either way a ``[jolt]`` line says which.

``python _topaz_train.py average <out.sav> <in1.sav> <in2.sav> ...``
    Uniform average of the floating-point parameters and buffers of several checkpoints of one run, saved as a
    whole Topaz model (usable by ``topaz extract`` and ``jolt detect_2d --weights``).
"""

import hashlib
import inspect
import os
import sys

# Run as a script, Python puts this file's directory (the jolt package) first on sys.path, where jolt's own
# topaz.py would shadow the Topaz package. Remove it before anything imports topaz (also needed to unpickle models).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

FAST_SAMPLER_TOPAZ_VERSION = "0.3.18"
#: SHA-256 of the source of topaz 0.3.18 MultipleImageSetDataset.__getitem__ (the method being replaced).
FAST_SAMPLER_STOCK_SHA256 = "c06e622a606a10cabc2cf9134ca8a92caa749a1ba0f74390449f66ebce2e810c"


def _fast_getitem(self, i):
    """Topaz 0.3.18 ``MultipleImageSetDataset.__getitem__`` with the positive-crop draw made O(1).

    Identical to the stock method except the marked line (stock: ``target = self.targets.sample()``).
    """
    import torchvision

    img_set_idx = self.rng.choice(len(self.paths), p=self.image_set_balance)
    if self.rng.random() < self.positive_balance:
        target = self.targets.iloc[[int(self.rng.integers(len(self.targets)))]]  # O(1); stock: targets.sample()
        name = target["image_name"].item()
        img = self.name_dict[name]
        y, x = target["y_coord"].item(), target["x_coord"].item()
        z = target["z_coord"].item() if self.dims == 3 else None
        crop, label = img.get_crop((z, y, x)), 1.0
    else:
        img_idx = self.rng.choice(len(self.paths[img_set_idx]))
        img = self.images[img_set_idx][img_idx]
        crop, label = img.get_UN_crop(), 0.0
    crop = crop.unsqueeze(0)
    if self.rotate:
        angle = self.rng.uniform(0, 360)
        crop = torchvision.transforms.functional.rotate(crop, angle)
        size_diff = crop.shape[-1] - self.crop_size
        xmin, xmax = size_diff // 2, size_diff // 2 + self.crop_size
        ymin, ymax = size_diff // 2, size_diff // 2 + self.crop_size
        crop = crop[..., ymin:ymax, xmin:xmax]
    if self.flip:
        if self.rng.random() < 0.5:
            crop = torchvision.transforms.functional.hflip(crop)
        if self.rng.random() < 0.5:
            crop = torchvision.transforms.functional.vflip(crop)
    crop = crop.squeeze(0)
    return crop, label


def apply_fast_sampler() -> str:
    """Replace the sampler if this is the expected Topaz; return a one-line status message."""
    import topaz
    from topaz.utils.data import memory_mapped_data as mmd

    cls = mmd.MultipleImageSetDataset
    if topaz.__version__ != FAST_SAMPLER_TOPAZ_VERSION:
        expected = FAST_SAMPLER_TOPAZ_VERSION
        return f"fast sampler not applied: Topaz {topaz.__version__} (expected {expected}); stock sampler"
    sha = hashlib.sha256(inspect.getsource(cls.__getitem__).encode()).hexdigest()
    if sha != FAST_SAMPLER_STOCK_SHA256:
        return f"fast sampler not applied: Topaz sampler source differs (sha256 {sha[:12]}); stock sampler"
    cls.__getitem__ = _fast_getitem
    return "fast sampler applied (O(1) positive-crop draw, same distribution)"


def average(out_path: str, paths: list[str]) -> None:
    """Average checkpoints of one Topaz run and save the result as a whole model."""
    import torch

    models = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    states = [m.state_dict() for m in models]
    keys = list(states[0])
    if any(list(s) != keys for s in states):
        sys.exit("checkpoints have different parameters; they must come from one training run")
    mean = {}
    for k in keys:
        t = states[-1][k]
        mean[k] = torch.stack([s[k].float() for s in states]).mean(0).to(t.dtype) if t.is_floating_point() else t
    model = models[-1]
    model.load_state_dict(mean)
    model.eval()
    torch.save(model, out_path)


def topaz_argv(train_arguments: list[str]) -> list[str]:
    """``sys.argv`` for Topaz's main: ``topaz train ...`` (``train`` added only if the arguments lack it)."""
    rest = list(train_arguments)
    return ["topaz", *rest] if rest[:1] == ["train"] else ["topaz", "train", *rest]


def main(argv: list[str]) -> int:
    """Dispatch ``train`` and ``average``."""
    if len(argv) >= 2 and argv[0] == "train":
        import random

        import numpy
        import torch

        seed = int(argv[1])
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[jolt] {apply_fast_sampler()}", flush=True)
        sys.argv = topaz_argv(argv[2:])
        from topaz.main import main as topaz_main

        return topaz_main()
    if len(argv) >= 3 and argv[0] == "average":
        average(argv[1], argv[2:])
        return 0
    if argv == ["which-topaz"]:  # diagnostics: where `import topaz` resolves from this script
        import importlib.util

        spec = importlib.util.find_spec("topaz")
        print(spec.origin if spec else "none")
        return 0
    sys.exit("usage: _topaz_train.py train <seed> <topaz train args> | average <out.sav> <in.sav> ... | which-topaz")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
