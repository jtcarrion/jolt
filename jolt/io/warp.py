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
    deselected_tilts : list of int
        1-based stack positions of the tilt images deselected in Warp (``UseTilt`` false in the ``.xml``); JOLT
        ignores their picks. Empty when every stack image is used.
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
    deselected_tilts: list[int] = field(default_factory=list)
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


def deselected_stack_tilts(
    name: str, xml_name: str, stack_angles: list[float], xml_angles: np.ndarray, use_tilt: np.ndarray
) -> tuple[list[int], list[str]]:
    """Find the stack images deselected in Warp; returns ``(1-based stack positions, warnings)``.

    Two layouts of a Warp ``.xml`` are accepted. Usually it lists every stack image, in stack order, and images
    deselected in Warp after the stack was written have ``UseTilt`` false. Otherwise the stack may hold only the used
    tilts, so the ``.xml`` has more entries than the stack, and nothing in the stack is deselected. When the ``.xml``
    order differs from the stack, images are matched by angle; a deselection that cannot be matched unambiguously
    is an error rather than a guess.

    Raises
    ------
    ValueError
        If neither layout fits, a deselected image cannot be identified, or every image is deselected.
    """
    angles = np.asarray(stack_angles, dtype=float)
    nz = len(angles)
    warnings: list[str] = []
    if len(xml_angles) == nz:
        order = np.arange(nz)
        if nz and float(np.max(np.abs(xml_angles - angles))) > ANGLE_TOLERANCE_DEG:
            nearest = np.array([int(np.argmin(np.abs(xml_angles - a))) for a in angles], dtype=int)
            unique = len(set(nearest.tolist())) == nz
            close = nz and float(np.max(np.abs(xml_angles[nearest] - angles))) <= ANGLE_TOLERANCE_DEG
            order = nearest if unique and close else None
        if order is None:
            if not use_tilt.all():
                raise ValueError(
                    f"{name}: {xml_name} deselects tilts (UseTilt) but its angles do not match the .tlt within "
                    f"{ANGLE_TOLERANCE_DEG} deg, so the deselected stack images cannot be identified"
                )
            worst = float(np.max(np.abs(np.sort(xml_angles) - np.sort(angles))))
            warnings.append(f"{name}: .xml and .tlt tilt angles differ by up to {worst:.2f} deg")
            return [], warnings
        deselected = [k + 1 for k in range(nz) if not use_tilt[order[k]]]
        if nz and len(deselected) == nz:
            raise ValueError(f"{name}: {xml_name} deselects every tilt (UseTilt)")
        return deselected, warnings
    if int(use_tilt.sum()) == nz:
        used_angles = xml_angles[use_tilt]
        worst = float(np.max(np.abs(np.sort(used_angles) - np.sort(angles)))) if nz else 0.0
        if worst > ANGLE_TOLERANCE_DEG:
            warnings.append(f"{name}: .xml and .tlt tilt angles differ by up to {worst:.2f} deg")
        return [], warnings
    raise ValueError(
        f"{name}: {xml_name} lists {len(xml_angles)} tilts ({int(use_tilt.sum())} used, UseTilt) but the stack has "
        f"{nz} images"
    )


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

    xml_ts = warpxmlfile.read(xml)
    deselected, warnings = deselected_stack_tilts(
        name, xml.name, angles, np.asarray(xml_ts.angles, dtype=float), np.asarray(xml_ts.use_tilt, dtype=bool)
    )
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
        deselected_tilts=deselected,
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

    The image dimensions set Warp's image center and its ``SizeRoundingFactors``, which the ``warp`` output frame
    needs (:func:`jolt.geometry.output_frame_transform`). Warp rewrites them from the first tilt's average before
    every reconstruction or export (``TiltSeries.LoadMovieData``), so the stored value is the real image size.

    The volume dimensions are only a cache: Warp overwrites them from the settings before every use, and
    ``ts_import_alignments`` stores a placeholder thickness of 1 A. Use :func:`read_settings` for the volume.

    A value that is all zeros (an XML written before Warp processed the tilt series) is returned as ``None``.
    """
    import warpxmlfile

    ts = warpxmlfile.read(Path(xml))
    image = getattr(ts, "image_dimensions_angstrom", None)
    volume = getattr(ts, "volume_dimensions_angstrom", None)
    image = tuple(float(v) for v in image) if image is not None else None
    volume = tuple(float(v) for v in volume) if volume is not None else None
    if image is not None and not any(image):
        image = None
    if volume is not None and not any(volume):
        volume = None
    return image, volume


@dataclass
class WarpSettings:
    """The values JOLT needs from a WarpTools ``.settings`` file.

    Attributes
    ----------
    path : Path
        The settings file.
    pixel_size : float
        ``Import.PixelSize`` (Angstrom), the pixel size Warp multiplies the tomogram dimensions by.
    tomo_dims_xyz : tuple of int
        ``Tomo.DimensionsX/Y/Z`` (pixels at ``pixel_size``).
    extension : str or None
        ``Import.Extension``: ``*.tomostar`` for tilt-series settings; frame-series settings name the movie files.
    """

    path: Path
    pixel_size: float
    tomo_dims_xyz: tuple[int, int, int]
    extension: str | None = None

    @property
    def is_tilt_series(self) -> bool:
        """Whether these are tilt-series settings (``Import.Extension`` ``*.tomostar``, or not given)."""
        return self.extension is None or self.extension.strip().lower().endswith(".tomostar")

    @property
    def volume_angstrom(self) -> tuple[float, float, float]:
        """Tomogram size in Angstrom, as Warp computes it (``ProcessingOptions.DimensionsPhysical``)."""
        return tuple(float(d) * self.pixel_size for d in self.tomo_dims_xyz)


def read_settings(path: str | PathLike, require_tilt_series: bool = True) -> WarpSettings:
    """Read ``Import.PixelSize`` and ``Tomo.DimensionsX/Y/Z`` from a WarpTools ``.settings`` file.

    WarpTools uses these, not the tilt-series XML, for the tomogram size in ``ts_reconstruct`` and
    ``ts_export_particles`` (``TomoProcessingOptionsBase.DimensionsPhysical = Dimensions * PixelSize``).

    A WarpTools project usually also has frame-series settings (``warp_frameseries.settings``, ``Import.Extension``
    the movie files), whose ``Tomo`` dimensions are unused defaults; with ``require_tilt_series`` they are refused.

    Raises
    ------
    ValueError
        If the file cannot be read as a WarpTools settings file, a value is missing or not positive, or (with
        ``require_tilt_series``) it holds frame-series settings.
    """
    import xml.etree.ElementTree as ET

    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"{path}: cannot read it as a WarpTools settings file ({exc})") from exc

    def param(section: str, name: str) -> float:
        node = root.find(f"./{section}/Param[@Name='{name}']")
        if node is None or node.get("Value") is None:
            raise ValueError(f"{path}: no {section}.{name}; is this a WarpTools settings file?")
        try:
            value = float(node.get("Value"))
        except ValueError as exc:
            raise ValueError(f"{path}: {section}.{name} is not a number: {node.get('Value')!r}") from exc
        if not value > 0:
            raise ValueError(f"{path}: {section}.{name} must be positive, got {value:g}")
        return value

    dims = tuple(param("Tomo", f"Dimensions{axis}") for axis in "XYZ")
    if any(d != int(d) for d in dims):
        raise ValueError(f"{path}: Tomo.DimensionsX/Y/Z must be whole pixels, got {dims}")
    ext_node = root.find("./Import/Param[@Name='Extension']")
    settings = WarpSettings(
        path=path,
        pixel_size=param("Import", "PixelSize"),
        tomo_dims_xyz=tuple(int(d) for d in dims),
        extension=ext_node.get("Value") if ext_node is not None else None,
    )
    if require_tilt_series and not settings.is_tilt_series:
        raise ValueError(
            f"{path}: these are frame-series settings (Import.Extension {settings.extension}); pass the tilt-series "
            "settings (Import.Extension *.tomostar) used for ts_reconstruct and ts_export_particles"
        )
    return settings


def find_settings(warp_dir: str | PathLike) -> tuple[WarpSettings | None, str]:
    """Find the WarpTools settings of a project; returns ``(settings or None, explanation)``.

    Looks for tilt-series ``*.settings`` directly in ``warp_dir`` (frame-series settings such as
    ``warp_frameseries.settings`` are skipped, see :func:`read_settings`). One file is used as is. Several files are
    accepted only if they all agree on ``Import.PixelSize`` and ``Tomo.DimensionsX/Y/Z`` (the values JOLT uses); then
    ``warp_tiltseries.settings`` is used if present, otherwise the first by name. ``None`` is returned when there
    is no tilt-series settings file, when they disagree, or when one cannot be read; the explanation says which.
    """
    candidates = sorted(Path(warp_dir).glob("*.settings"))
    found, skipped, problems = [], [], []
    for path in candidates:
        try:
            s = read_settings(path, require_tilt_series=False)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        (found if s.is_tilt_series else skipped).append(s)
    if problems:
        return None, "cannot read every settings file:\n  " + "\n  ".join(problems)
    note = f"; skipped frame-series {', '.join(s.path.name for s in skipped)}" if skipped else ""
    if not found:
        return None, f"no tilt-series *.settings file in {warp_dir}{note}"
    if len({(s.pixel_size, s.tomo_dims_xyz) for s in found}) > 1:
        listed = "\n  ".join(f"{s.path.name}: PixelSize {s.pixel_size:g}, Dimensions {s.tomo_dims_xyz}" for s in found)
        return None, f"{len(found)} tilt-series settings files in {warp_dir} disagree:\n  {listed}"
    chosen = next((s for s in found if s.path.name == "warp_tiltseries.settings"), found[0])
    if len(found) == 1:
        return chosen, f"{chosen.path.name} (the only tilt-series settings file in {warp_dir}{note})"
    return chosen, (
        f"{chosen.path.name} ({len(found)} tilt-series settings files in {warp_dir}, all with the same tomogram "
        f"size{note})"
    )


def settings_volume_check(
    settings: WarpSettings, tomo_dims_xyz: tuple[int, int, int], stack_pixel_size: float, tol_px: float = 2.0
) -> str | None:
    """Compare the settings' tomogram size with ``tomo_dims_xyz`` (pixels at ``stack_pixel_size``).

    Returns a warning if any axis differs by more than ``tol_px`` pixels (the settings probably belong to other
    tomograms), otherwise ``None``.
    """
    expected = np.asarray(settings.volume_angstrom) / float(stack_pixel_size)
    if np.any(np.abs(expected - np.asarray(tomo_dims_xyz, dtype=float)) > tol_px):
        shown = ", ".join(f"{v:.2f}" for v in expected)
        return (
            f"{settings.path.name} gives a tomogram of ({shown}) px at {stack_pixel_size:g} A/px but --tomo_dims_xyz "
            f"is {tuple(int(v) for v in tomo_dims_xyz)}; check that the settings belong to these tomograms"
        )
    return None


def settings_record(settings: WarpSettings) -> dict:
    """Return the ``project.warp_settings`` entry of ``jolt.yaml`` for ``settings`` (path and the values used)."""
    return {
        "path": str(Path(settings.path).resolve()),
        "pixel_size": float(settings.pixel_size),
        "tomo_dims_xyz": [int(v) for v in settings.tomo_dims_xyz],
    }


def settings_from_record(record: dict) -> WarpSettings:
    """Return the :class:`WarpSettings` of a ``project.warp_settings`` entry (inverse of :func:`settings_record`)."""
    return WarpSettings(
        path=Path(record["path"]),
        pixel_size=float(record["pixel_size"]),
        tomo_dims_xyz=tuple(int(v) for v in record["tomo_dims_xyz"]),
    )


def resolve_project_settings(project: dict, override: str | PathLike | None = None) -> tuple[WarpSettings, str, bool]:
    """WarpTools settings for a JOLT project; returns ``(settings, explanation, new)``.

    ``override`` (a command's ``--warp_settings``) wins; then ``project.warp_settings`` recorded by ``jolt import``;
    then a search of ``project.warp_dir`` (:func:`find_settings`), for projects imported before JOLT recorded the
    settings. ``new`` is True when the result is not yet recorded in ``jolt.yaml`` and should be saved.

    Raises
    ------
    ValueError
        If no settings can be found unambiguously; the message says how to provide them.
    """
    if override is not None:
        settings = read_settings(override)
        recorded = project.get("warp_settings")
        new = recorded is None or settings_record(settings) != recorded
        return settings, f"{settings.path} (--warp_settings)", new
    if project.get("warp_settings"):
        settings = settings_from_record(project["warp_settings"])
        return settings, f"{settings.path} (recorded in jolt.yaml)", False
    settings, why = find_settings(project["warp_dir"])
    if settings is None:
        raise ValueError(
            "the warp output frame needs the WarpTools settings file used for ts_reconstruct / ts_export_particles "
            f"(its Tomo dimensions and pixel size set the tomogram center), and none was found: {why}. Pass it with "
            "--warp_settings <file>.settings, or use --output_frame teamtomo"
        )
    return settings, f"{why}; not recorded in jolt.yaml before this run", True
