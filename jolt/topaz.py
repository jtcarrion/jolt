"""Run Topaz (preprocess, extract, train) as an external program.

Topaz runs in its own environment and is called as a command, so JOLT never imports Topaz or its pinned
PyTorch. Runs are made reproducible by seeding numpy inside Topaz's own Python interpreter: Topaz fits its
preprocessing mixture model on a random pixel subset, and unseeded runs pick about 20% different particles.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import mrcfile
import numpy as np

#: Pretrained Topaz models that can be named instead of a ``.sav`` file.
PRETRAINED_MODELS = ("resnet8", "resnet16", "resnet8_u32", "resnet8_u64", "resnet16_u32", "resnet16_u64")
DEFAULT_MODEL = "resnet16"
DEFAULT_THRESHOLD = -6.0
#: With ``--top_k`` every non-maximum-suppression peak is extracted and the best K are kept afterwards.
TOP_K_THRESHOLD_FLOOR = -1.0e6
#: Topaz preprocess fits on one pixel in this many (``--sample``); fitting on all pixels can collapse.
PREPROCESS_SAMPLE = 10
_SEEDED_TOPAZ = (
    "import sys, numpy; numpy.random.seed(int(sys.argv[1])); "
    "sys.argv = ['topaz'] + sys.argv[2:]; "
    "from topaz.main import main; sys.exit(main())"
)


@dataclass
class TopazInstall:
    """A Topaz executable, its version, and the Python interpreter of its environment (for seeding)."""

    executable: str
    version: str
    python: str | None

    def command(self, args: list[str], seed: int) -> list[str]:
        """Return the command line for ``topaz <args>``, seeded when Topaz's interpreter is known."""
        if self.python:
            return [self.python, "-c", _SEEDED_TOPAZ, str(seed), *args]
        return [self.executable, *args]


def _topaz_python(executable: str) -> str | None:
    candidates = [str(Path(executable).parent / "python"), str(Path(executable).parent / "python3")]
    try:
        with open(executable) as handle:
            for line in handle.read(2000).splitlines()[:4]:
                for token in line.replace("'", " ").replace("#!", " ").split():
                    if "python" in Path(token).name:
                        candidates.append(token)
    except (OSError, UnicodeDecodeError):
        pass
    for python in candidates:
        if os.path.isfile(python) and os.access(python, os.X_OK):
            if subprocess.run([python, "-c", "import topaz.main"], capture_output=True, timeout=300).returncode == 0:
                return python
    return None


def find_topaz(topaz_bin: str | None = None) -> TopazInstall:
    """Locate Topaz: ``topaz_bin``, then ``$JOLT_TOPAZ_BIN``, then ``topaz`` on the PATH.

    Raises
    ------
    FileNotFoundError, RuntimeError
        If Topaz cannot be found or does not run.
    """
    candidate = topaz_bin or os.environ.get("JOLT_TOPAZ_BIN") or shutil.which("topaz")
    if not candidate:
        raise FileNotFoundError("Topaz not found; pass --topaz_bin /path/to/envs/topaz/bin/topaz or set JOLT_TOPAZ_BIN")
    resolved = shutil.which(candidate) or candidate
    if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
        raise FileNotFoundError(f"Topaz executable not found or not executable: {candidate}")
    out = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=300)
    lines = (out.stdout or out.stderr).strip().splitlines()
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"'{resolved} --version' failed:\n{out.stderr.strip()}")
    return TopazInstall(resolved, lines[-1], _topaz_python(resolved))


def resolve_model(weights: str | None) -> str:
    """Return a pretrained model name or the absolute path of a ``.sav`` file."""
    model = weights or DEFAULT_MODEL
    if model in PRETRAINED_MODELS:
        return model
    path = Path(model).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"weights must be a .sav file or one of {PRETRAINED_MODELS}: {model}")
    return str(path)


def tilt_image_name(index0: int, n_tilts: int) -> str:
    """Name of the tilt image at 0-based stack position ``index0`` (``tilt_01`` ..., 1-based)."""
    return f"tilt_{index0 + 1:0{max(2, len(str(n_tilts)))}d}"


def split_stack(stack: str | PathLike, out_dir: str | PathLike) -> list[Path]:
    """Write every image of a stack to ``out_dir/tilt_NN.mrc`` (32-bit float, pixel size kept)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with mrcfile.mmap(stack, mode="r", permissive=True) as mrc:
        data = mrc.data if mrc.data.ndim == 3 else mrc.data[np.newaxis]
        voxel_size = mrc.voxel_size
        for i in range(data.shape[0]):
            path = out_dir / f"{tilt_image_name(i, data.shape[0])}.mrc"
            with mrcfile.new(path, overwrite=True) as out:
                out.set_data(np.asarray(data[i], dtype=np.float32))
                out.voxel_size = voxel_size
            paths.append(path)
    return paths


def run_logged(command: list[str], log_path: str | PathLike) -> None:
    """Run a command, appending the command line and all its output to ``log_path``; raise if it fails."""
    with open(log_path, "a") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def preprocess(topaz: TopazInstall, images: list[Path], out_dir: Path, device: int, seed: int, log_path) -> list[Path]:
    """``topaz preprocess -s 1 --sample 10`` (no downsampling); returns the preprocessed images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_logged(
        topaz.command(
            ["preprocess", "-s", "1", "--sample", str(PREPROCESS_SAMPLE), f"--device={device}", "-o", str(out_dir)]
            + [str(p) for p in images],
            seed,
        ),
        log_path,
    )
    out = [out_dir / p.name for p in images]
    missing = [p.name for p in out if not p.exists()]
    if missing:
        raise RuntimeError(f"topaz preprocess did not write: {', '.join(missing[:5])}")
    return out


def extract(
    topaz: TopazInstall,
    images: list[Path],
    model: str,
    radius: int,
    threshold: float,
    device: int,
    seed: int,
    out_path: Path,
    log_path,
) -> dict[str, list[tuple]]:
    """``topaz extract``; returns ``{image stem: [(x, y, score), ...]}`` with x, y as Topaz wrote them."""
    run_logged(
        topaz.command(
            [
                "extract",
                "-m",
                model,
                "-r",
                str(radius),
                f"--threshold={threshold}",
                f"--device={device}",
                "-o",
                str(out_path),
            ]
            + [str(p) for p in images],
            seed,
        ),
        log_path,
    )
    picks: dict[str, list[tuple]] = {}
    with open(out_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            picks.setdefault(row["image_name"], []).append((row["x_coord"], row["y_coord"], float(row["score"])))
    return picks


def write_picks_star(path: str | PathLike, rows: list[tuple]) -> None:
    """Write picks ``(image, x, y, score, tilt series)`` as a STAR loop (atomically).

    Coordinates are written exactly as Topaz reported them; the score with six decimals.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        handle.write("\ndata_\n\nloop_\n")
        for i, column in enumerate(
            ("_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY", "_rlnAutopickFigureOfMerit", "_TomoName"), 1
        ):
            handle.write(f"{column} #{i}\n")
        for image, x, y, score, tomo in rows:
            handle.write(f"{image}\t{x}\t{y}\t{score:.6f}\t{tomo}\n")
    os.replace(tmp, path)


def detect_tilt_series(
    name: str,
    stack: str | PathLike,
    n_tilts: int,
    out_star: str | PathLike,
    *,
    topaz: TopazInstall,
    model: str,
    radius: int,
    threshold: float | None,
    top_k: int | None,
    seed: int,
    device: int,
    work_dir: Path,
    tilts_dir: Path | None,
    keep_work: bool = False,
    log=print,
) -> dict:
    """Pick 2D particles on every raw tilt image of one tilt series and write the picks STAR.

    Parameters
    ----------
    name : str
        Tilt-series name (written to ``_TomoName``).
    stack : path-like
        Raw tilt stack.
    n_tilts : int
        Number of images (checked against the stack).
    out_star : path-like
        Output STAR file. Tilt images are named ``tilt_NN`` (1-based stack position).
    topaz : TopazInstall
    model : str
        Pretrained model name or ``.sav`` path.
    radius : int
        Extraction (non-maximum suppression) radius in pixels.
    threshold, top_k :
        Keep picks scoring above ``threshold``, or the best ``top_k`` per tilt image (exactly one is set).
    seed : int
        Seed for Topaz preprocessing.
    device : int
        GPU index as Topaz numbers them (``-1`` = CPU).
    work_dir : pathlib.Path
        Scratch directory for this tilt series (removed on success unless ``keep_work``).
    tilts_dir : pathlib.Path or None
        Where to keep the preprocessed tilt images Topaz picked on (``None``: not kept).
    log : callable

    Returns
    -------
    dict
        Pick statistics of the tilt series.
    """
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    log_path = work_dir / "topaz.log"
    images = split_stack(stack, work_dir / "split")
    if len(images) != n_tilts:
        raise ValueError(f"{name}: the stack has {len(images)} images but the .tlt has {n_tilts} angles")
    pp_dir = tilts_dir or work_dir / "preprocessed"
    if tilts_dir is not None and tilts_dir.exists():
        for old in tilts_dir.glob("tilt_*.mrc"):
            old.unlink()
    pp_images = preprocess(topaz, images, pp_dir, device, seed, log_path)
    if not keep_work:
        shutil.rmtree(work_dir / "split", ignore_errors=True)
    cut = TOP_K_THRESHOLD_FLOOR if top_k is not None else threshold
    picks = extract(topaz, pp_images, model, radius, cut, device, seed, work_dir / "topaz_extract.txt", log_path)
    expected = {p.stem for p in pp_images}
    unknown = set(picks) - expected
    if unknown:
        raise RuntimeError(f"{name}: unexpected image names in the Topaz output: {sorted(unknown)[:5]}")
    rows = []
    for image in sorted(expected):
        tilt_picks = sorted(picks.get(image, []), key=lambda t: -t[2])
        if top_k is not None:
            tilt_picks = tilt_picks[:top_k]
        rows.extend((image, x, y, s, name) for x, y, s in tilt_picks)
    Path(out_star).parent.mkdir(parents=True, exist_ok=True)
    write_picks_star(out_star, rows)
    counts = [sum(1 for r in rows if r[0] == image) for image in sorted(expected)]
    stats = {
        "n_tilts": n_tilts,
        "total_picks": len(rows),
        "avg_picks_per_tilt": round(len(rows) / n_tilts, 2) if n_tilts else 0,
        "min_picks_per_tilt": int(min(counts)) if counts else 0,
        "max_picks_per_tilt": int(max(counts)) if counts else 0,
        "n_tilts_without_picks": int(sum(c == 0 for c in counts)),
        "tilts_dir": str(tilts_dir) if tilts_dir else None,
        "seeded": topaz.python is not None,
    }
    log(f"[{name}] {len(rows)} picks ({stats['avg_picks_per_tilt']} per tilt image) -> {out_star}")
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return stats
