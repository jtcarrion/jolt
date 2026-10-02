"""JOLT: 3D particle picking from 2D tilt-image picks with joint tilt-series alignment refinement."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jolt-cryoet")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
