"""Discover tilt series in a WarpTools project and check that their files agree with each other.

A WarpTools project has one ``<name>.xml`` per tilt series in ``warp_tiltseries/`` and one folder per tilt
series in ``warp_tiltseries/tiltstack/<name>/`` holding the raw stack ``<name>.st``, the IMOD alignment
``<name>.xf`` and the tilt angles ``<name>.tlt`` (and usually the aligned stack ``<name>_ali.mrc``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np

from jolt.io.imod import read_tlt, read_xf
from jolt.io.mrc import read_header_dims, read_pixel_size

#: Largest difference (degrees) between Warp .xml angles and .tlt angles accepted without a warning.
ANGLE_TOLERANCE_DEG = 0.5


@dataclass
class TiltSeries:
    """The files and checked properties of one tilt series.

    Attributes
    ----------
    name : str
        Tilt-series name, e.g. ``TS_001``.
    xml : Path
        Warp ``.xml`` file.
    raw_stack : Path
        Raw (unaligned) tilt stack ``.st``.
    xf, tlt : Path
        IMOD alignment and tilt-angle files.
    aligned_stack : Path or None
        Aligned stack (``<name>_ali.mrc``) if present.
    raw_dims_xy : tuple of int
        Raw tilt-image width and height in pixels (from the ``.st`` header).
    n_tilts : int
        Number of tilt images in the stack (== ``.tlt`` lines == ``.xf`` rows).
    pixel_size : float
        Tilt-stack pixel size in Angstrom (from the ``.st`` header).
    tilt_angles : list of float
        Angles from the ``.tlt``, in stack order.
    warnings : list of str
        Non-fatal inconsistencies found while checking.
    """

    name: str
    xml: Path
    raw_stack: Path
    xf: Path
    tlt: Path
    aligned_stack: Path | None
    raw_dims_xy: tuple[int, int]
    n_tilts: int
    pixel_size: float
    tilt_angles: list[float]
    warnings: list[str] = field(default_factory=list)


def default_paths(warp_dir: str | PathLike) -> tuple[Path, Path]:
    """Return the default ``(xml_path, tiltstack_path)`` of a WarpTools project."""
    root = Path(warp_dir) / "warp_tiltseries"
    return root, root / "tiltstack"


def _first_existing(folder: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = folder / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_tilt_series_names(xml_path: Path, tiltstack_path: Path) -> list[str]:
    """Return the sorted names that have both a ``<name>.xml`` and a ``tiltstack/<name>/`` folder."""
    xml_names = {p.stem for p in xml_path.glob("*.xml")}
    folder_names = {p.name for p in tiltstack_path.iterdir() if p.is_dir()} if tiltstack_path.is_dir() else set()
    return sorted(xml_names & folder_names)


def check_tilt_series(name: str, xml_path: Path, tiltstack_path: Path) -> TiltSeries:
    """Find and cross-check the files of one tilt series.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    ValueError
        If the stack, ``.tlt``, ``.xf`` and ``.xml`` disagree on the number of tilts.
    """
    import warpxmlfile

    folder = tiltstack_path / name
    xml = xml_path / f"{name}.xml"
    raw_stack = _first_existing(folder, [f"{name}.st"])
    xf = _first_existing(folder, [f"{name}.xf", f"{name}_fid.xf"])
    tlt = _first_existing(folder, [f"{name}.tlt", f"{name}_fid.tlt", f"{name}.rawtlt"])
    missing = [label for label, path in (("raw stack .st", raw_stack), (".xf", xf), (".tlt", tlt)) if path is None]
    if missing or not xml.is_file():
        missing += [] if xml.is_file() else [".xml"]
        raise FileNotFoundError(f"{name}: missing {', '.join(missing)} (looked in {folder} and {xml_path})")
    aligned_stack = _first_existing(folder, [f"{name}_ali.mrc"])

    nx, ny, nz = read_header_dims(raw_stack)
    angles = read_tlt(tlt)
    read_xf(xf, angles)  # raises if the row count differs from the angle count
    if nz != len(angles):
        raise ValueError(f"{name}: {raw_stack.name} has {nz} images but {tlt.name} has {len(angles)} angles")

    warnings = []
    xml_ts = warpxmlfile.read(xml)
    used = np.asarray(xml_ts.use_tilt, dtype=bool)
    xml_angles = np.asarray(xml_ts.angles, dtype=float)[used]
    if len(xml_angles) != nz:
        raise ValueError(
            f"{name}: {xml.name} marks {len(xml_angles)} tilts as used (UseTilt) but the stack has {nz} images"
        )
    worst = float(np.max(np.abs(np.sort(xml_angles) - np.sort(angles)))) if nz else 0.0
    if worst > ANGLE_TOLERANCE_DEG:
        warnings.append(f"{name}: .xml and .tlt tilt angles differ by up to {worst:.2f} deg")
    if aligned_stack is not None:
        anx, any_, anz = read_header_dims(aligned_stack)
        if anz != nz:
            raise ValueError(f"{name}: {aligned_stack.name} has {anz} images but the raw stack has {nz}")

    return TiltSeries(
        name=name,
        xml=xml,
        raw_stack=raw_stack,
        xf=xf,
        tlt=tlt,
        aligned_stack=aligned_stack,
        raw_dims_xy=(nx, ny),
        n_tilts=nz,
        pixel_size=read_pixel_size(raw_stack),
        tilt_angles=angles,
        warnings=warnings,
    )


def discover(
    warp_dir: str | PathLike,
    xml_path: str | PathLike | None = None,
    tiltstack_path: str | PathLike | None = None,
    names: list[str] | None = None,
) -> list[TiltSeries]:
    """Find and check every tilt series of a WarpTools project.

    Parameters
    ----------
    warp_dir : str or path-like
        WarpTools project directory.
    xml_path, tiltstack_path : str or path-like, optional
        Overrides for ``<warp_dir>/warp_tiltseries`` and ``<warp_dir>/warp_tiltseries/tiltstack``.
    names : list of str, optional
        Only these tilt series (each must exist).

    Returns
    -------
    list of TiltSeries
        Sorted by name.

    Raises
    ------
    FileNotFoundError, ValueError
        If no tilt series is found or any fails a check; problems of all tilt series are reported together.
    """
    default_xml, default_stack = default_paths(warp_dir)
    xml_path = Path(xml_path) if xml_path else default_xml
    tiltstack_path = Path(tiltstack_path) if tiltstack_path else default_stack
    found = find_tilt_series_names(xml_path, tiltstack_path)
    if names:
        unknown = sorted(set(names) - set(found))
        if unknown:
            raise FileNotFoundError(f"tilt series not found in {xml_path} / {tiltstack_path}: {unknown}")
        found = [n for n in found if n in set(names)]
    if not found:
        raise FileNotFoundError(f"no tilt series with both a .xml in {xml_path} and a folder in {tiltstack_path}")
    result, errors = [], []
    for name in found:
        try:
            result.append(check_tilt_series(name, xml_path, tiltstack_path))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(f"{len(errors)} of {len(found)} tilt series failed the checks:\n  " + "\n  ".join(errors))
    return result


def physical_dimensions(xml: str | PathLike) -> tuple[tuple[float, float] | None, tuple[float, float, float] | None]:
    """``ImageDimensionsAngstrom`` and ``DimensionsAngstrom`` of a Warp tilt-series XML (``None`` if absent).

    They define Warp's image and volume centers and its ``SizeRoundingFactors``, which the ``warp`` output frame
    needs (:func:`jolt.geometry.output_frame_transform`).
    """
    import warpxmlfile

    ts = warpxmlfile.read(Path(xml))
    image = getattr(ts, "image_dimensions_angstrom", None)
    volume = getattr(ts, "volume_dimensions_angstrom", None)
    image = tuple(float(v) for v in image) if image is not None else None
    volume = tuple(float(v) for v in volume) if volume is not None else None
    return image, volume
