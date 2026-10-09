"""Run Topaz (preprocess, extract, train) as an external program.

Topaz runs in its own environment and is called as a command, so JOLT never imports Topaz or its pinned
PyTorch. Runs are made reproducible by seeding numpy inside Topaz's own Python interpreter: Topaz fits its
preprocessing mixture model on a random pixel subset, and unseeded runs pick about 20% different particles.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import mrcfile
import numpy as np

from jolt.io.imod import write_picks_mod

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
    # The same picks as an IMOD model on the raw stack (3dmod <stack> <model>); stack position = Topaz image order.
    out_mod = Path(out_star).with_suffix(".mod")
    section_of = {p.stem: i for i, p in enumerate(pp_images)}
    by_section: dict[int, list] = {}
    for image, x, y, _, _ in rows:
        by_section.setdefault(section_of[image], []).append((float(x), float(y)))
    if by_section:
        with mrcfile.open(stack, header_only=True, permissive=True) as mrc:
            dims_xy = (int(mrc.header.nx), int(mrc.header.ny))
        write_picks_mod(
            out_mod, {k: np.array(v) for k, v in by_section.items()}, dims_xy, n_tilts, sphere_radius=radius
        )
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
    stats["picks_mod"] = str(out_mod) if by_section else None
    log(
        f"[{name}] {len(rows)} picks ({stats['avg_picks_per_tilt']} per tilt image) -> {out_star}"
        + (" (+ .mod)" if by_section else "")
    )
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return stats


_SEEDED_TOPAZ_TRAIN = (
    "import sys, random, numpy, torch; s = int(sys.argv[1]); random.seed(s); "
    "numpy.random.seed(s); torch.manual_seed(s); sys.argv = ['topaz'] + sys.argv[2:]; "
    "from topaz.main import main; sys.exit(main())"
)
_P_OBSERVED = re.compile(r"^#?\s*(\S+)\s+(train|test)\s+([0-9.eE+-]+)\s+(\d+)\s+(\d+)\s*$")


def dedup_labels(xy: np.ndarray, radius: float) -> np.ndarray:
    """Greedy mask keeping a label only if it is at least ``radius`` pixels from every label kept before it."""
    if radius <= 0 or len(xy) < 2:
        return np.ones(len(xy), bool)
    keep = np.zeros(len(xy), bool)
    kept: list[np.ndarray] = []
    for i, p in enumerate(xy):
        if kept and (np.hypot(*(np.asarray(kept) - p).T) < radius).any():
            continue
        keep[i] = True
        kept.append(p)
    return keep


def write_training_files(images: list[tuple[str, Path, np.ndarray, str]], out_dir: Path, radius: int) -> dict:
    """Write Topaz training tables.

    Parameters
    ----------
    images : list of tuple
        ``(image name, preprocessed image path, (n, 2) labels, role)`` with role ``"train"`` or ``"test"``.
        Image names must be unique; the files are linked under that name (Topaz matches labels to images by
        the file's base name).
    out_dir : pathlib.Path
    radius : int
        Label radius; labels closer than this are de-duplicated.

    Returns
    -------
    dict
        Image and label counts per role.
    """
    links = out_dir / "images"
    links.mkdir(parents=True, exist_ok=True)
    counts = {}
    for role in ("train", "test"):
        image_lines, coord_lines, n_raw, n_kept = [], [], 0, 0
        for name, path, xy, r in images:
            if r != role:
                continue
            link = links / f"{name}.mrc"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(path).resolve())
            image_lines.append(f"{name}\t{link}")
            keep = dedup_labels(xy, radius)
            n_raw += len(keep)
            n_kept += int(keep.sum())
            coord_lines += [f"{name}\t{x:.2f}\t{y:.2f}" for x, y in xy[keep]]
        (out_dir / f"{role}_images.txt").write_text("image_name\tpath\n" + "\n".join(image_lines) + "\n")
        (out_dir / f"{role}_coords.txt").write_text("image_name\tx_coord\ty_coord\n" + "\n".join(coord_lines) + "\n")
        counts[role] = {"n_images": len(image_lines), "labels_raw": n_raw, "labels_kept": n_kept}
    return counts


def train(
    topaz: TopazInstall,
    data_dir: Path,
    run_dir: Path,
    *,
    model: str,
    radius: int,
    num_particles: float,
    learning_rate: float,
    minibatch_size: int,
    epoch_size: int,
    epochs: int,
    seed: int,
    device: int,
    log=print,
) -> dict:
    """Run ``topaz train`` (GE-binomial, from a pretrained model) on tables made by :func:`write_training_files`.

    Training stops at once if Topaz reports that no labeled region was observed on the training split (a
    sign that image names and label names do not match).

    Returns
    -------
    dict
        ``model`` (path of the final epoch's model, or None), ``p_observed``, ``returncode``, ``aborted`` and
        the held-out precision curve if Topaz wrote one.
    """
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    args = [
        "train",
        "--train-images", str(data_dir / "train_images.txt"), "--train-targets", str(data_dir / "train_coords.txt"),
        "--test-images", str(data_dir / "test_images.txt"), "--test-targets", str(data_dir / "test_coords.txt"),
        "-n", f"{num_particles:g}", "-r", str(radius), "-m", model, "--pretrained", "--method", "GE-binomial",
        "--learning-rate", f"{learning_rate:g}", "--minibatch-size", str(minibatch_size), "--epoch-size",
        str(epoch_size), "--num-epochs", str(epochs), f"--device={device}",
        "--save-prefix", str(run_dir / "models" / "model"), "-o", str(run_dir / "train_curve.txt"),
    ]  # fmt: skip
    command = [topaz.python, "-c", _SEEDED_TOPAZ_TRAIN, str(seed), *args] if topaz.python else [topaz.executable, *args]
    info: dict = {"p_observed": {}, "warnings": [], "aborted": None, "command": " ".join(command)}
    with open(run_dir / "topaz_train.log", "w") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            m = _P_OBSERVED.match(line.strip())
            if m and m.group(1) != "source":
                split, p_obs = m.group(2), float(m.group(3))
                info["p_observed"][split] = p_obs
                log(f"p_observed[{split}] = {p_obs:.4e} ({m.group(4)} of {m.group(5)} regions)")
                if split == "train" and p_obs == 0.0:
                    info["aborted"] = "p_observed = 0 on the training split (image names do not match the labels)"
                    proc.kill()
                    break
            if line.startswith("WARNING"):
                info["warnings"].append(line.strip())
        proc.wait()
    info["returncode"] = proc.returncode
    final = run_dir / "models" / f"model_epoch{epochs}.sav"
    info["model"] = str(final) if final.exists() and not info["aborted"] and proc.returncode == 0 else None
    curve = run_dir / "train_curve.txt"
    if curve.exists():
        lines = [
            x.rstrip("\n").split("\t") for x in curve.read_text().splitlines() if x.strip() and not x.startswith("#")
        ]
        if lines:
            head = lines[0]
            if "split" in head and "auprc" in head:
                info["test_auprc_by_epoch"] = [
                    float(r[head.index("auprc")]) for r in lines[1:]
                    if len(r) == len(head) and r[head.index("split")] == "test" and r[head.index("auprc")] != "-"
                ]  # fmt: skip
    return info
