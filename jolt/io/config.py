"""Read, write and validate the project configuration file ``jolt.yaml``.

``jolt import`` creates the file; every later command reads it and records its own section (the parameters
it actually used and where its outputs went). The file records inputs and settings only. It never stores
git or environment information.

Layout::

    config_version: 1
    project:
      warp_dir: /path/to/warp_project
      xml_path: /path/to/warp_project/warp_tiltseries
      tiltstack_path: /path/to/warp_project/warp_tiltseries/tiltstack
      tomo_dims_xyz: [630, 630, 200]          # pixels at the tilt-stack pixel size
    tilt_series:
      TS_5:
        xml: ...
        raw_stack: ...
        aligned_stack: null
        xf: ...
        tlt: ...
        n_tilts: 31
        deselected_tilts: []                 # 1-based stack positions deselected in Warp (UseTilt); optional
        raw_dims_xy: [630, 630]
        pixel_size: 10.0
        picks_2d: ...                        # optional until detect_2d or import --picks_2d
        picks_frame: raw
    commands:
      import: {...}
      joint_optimize: {...}
"""

from __future__ import annotations

import datetime
from os import PathLike
from pathlib import Path

import yaml

CONFIG_NAME = "jolt.yaml"
CONFIG_VERSION = 1
PICKS_FRAMES = ("raw", "aligned")


def config_path(project_dir: str | PathLike) -> Path:
    """Return the path of ``jolt.yaml`` inside a project directory."""
    return Path(project_dir) / CONFIG_NAME


def validate(config: dict, source: str = CONFIG_NAME) -> None:
    """Check the structure of a loaded configuration.

    Raises
    ------
    ValueError
        With every problem found.
    """
    problems = []
    if config.get("config_version") != CONFIG_VERSION:
        problems.append(f"config_version is {config.get('config_version')!r}, expected {CONFIG_VERSION}")
    project = config.get("project") or {}
    dims = project.get("tomo_dims_xyz")
    if not (isinstance(dims, list) and len(dims) == 3 and all(isinstance(d, int) and d > 0 for d in dims)):
        problems.append(f"project.tomo_dims_xyz must be three positive integers, got {dims!r}")
    series = config.get("tilt_series")
    if not isinstance(series, dict) or not series:
        problems.append("tilt_series is empty")
        series = {}
    for name, entry in series.items():
        for key in ("raw_stack", "xf", "tlt", "n_tilts", "raw_dims_xy", "pixel_size"):
            if entry.get(key) is None:
                problems.append(f"tilt_series.{name}.{key} is missing")
        deselected = entry.get("deselected_tilts", [])
        n_tilts = entry.get("n_tilts")
        if not isinstance(deselected, list) or not all(
            isinstance(i, int) and (not isinstance(n_tilts, int) or 1 <= i <= n_tilts) for i in deselected
        ):
            problems.append(f"tilt_series.{name}.deselected_tilts must list stack positions 1..n_tilts")
        frame = entry.get("picks_frame", "raw")
        if frame not in PICKS_FRAMES:
            problems.append(f"tilt_series.{name}.picks_frame is {frame!r}, expected one of {PICKS_FRAMES}")
    if problems:
        raise ValueError(f"{source}: invalid configuration:\n  " + "\n  ".join(problems))


def load(project_dir: str | PathLike) -> dict:
    """Load and validate ``<project_dir>/jolt.yaml``."""
    path = config_path(project_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; create the project with `jolt import` first")
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    validate(config, str(path))
    return config


def save(project_dir: str | PathLike, config: dict) -> Path:
    """Validate and write ``<project_dir>/jolt.yaml`` (atomically, via a temporary file)."""
    validate(config)
    path = config_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=None)
    tmp.replace(path)
    return path


def record_command(config: dict, command: str, parameters: dict, outputs: dict | None = None) -> None:
    """Record the parameters a command used (and its outputs) in ``config['commands'][command]``.

    Only plain values (numbers, strings, booleans, lists, dictionaries, paths) are stored; paths are stored
    as strings. The time of the run is recorded, nothing about the software environment.
    """

    def plain(value):
        if isinstance(value, PathLike):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        return value

    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "parameters": plain(parameters)}
    if outputs:
        entry["outputs"] = plain(outputs)
    config.setdefault("commands", {})[command] = entry
