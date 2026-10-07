"""Small shared helpers: thread limits, logging to a file, device choice, tilt-series lists."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

#: Environment variables that set the number of CPU math threads of numpy, scipy and torch.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def set_thread_limits(n_threads: int) -> None:
    """Limit CPU math threads (OpenMP, MKL, OpenBLAS) in this process and in processes it starts.

    Call before numpy or torch do any work. Many concurrent workers each using one thread per core produced
    wrong results with two OpenMP runtimes loaded and are much slower; one thread per worker is the default.
    """
    n = str(max(1, int(n_threads)))
    for var in THREAD_ENV_VARS:
        os.environ[var] = n
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(int(n))
    except ImportError:  # threadpoolctl is a dependency; this only guards unusual installs
        pass
    try:
        import torch

        torch.set_num_threads(int(n))
    except ImportError:
        pass


def parse_name_list(text: str | None) -> list[str] | None:
    """Split a comma-separated list of names; ``None`` or empty means "all"."""
    if not text:
        return None
    names = [part.strip() for part in text.split(",") if part.strip()]
    return names or None


class FileLog:
    """Callable that prints a message and appends it to a log file."""

    def __init__(self, path: str | os.PathLike, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def __call__(self, message: str) -> None:
        """Print and append one message."""
        if self.echo:
            print(message, flush=True)
        with open(self.path, "a") as handle:
            handle.write(message + "\n")


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show an option's default only when it is informative and the help text does not already state one.

    Flags (``False``) and options without a default (``None``) are shown without "(default: ...)".
    """

    def _get_help_string(self, action):
        text = action.help or ""
        if "default" in text or action.default in (None, False, argparse.SUPPRESS) or not action.option_strings:
            return text
        return super()._get_help_string(action)
