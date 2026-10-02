"""Read and write RELION-style STAR files (wrapper around ``starfile``).

Tilt-series, tomogram and micrograph names are always read as strings (``TS_01`` must not become a number),
column names carry no leading underscore in memory, and files are written tab-separated.
"""

from __future__ import annotations

from os import PathLike

import pandas as pd
import starfile

#: Columns that hold names and must never be parsed as numbers.
NAME_COLUMNS = ["rlnMicrographName", "rlnTomoName", "rlnImageName", "rlnTomoParticleName"]


def read_star(path: str | PathLike) -> dict[str, pd.DataFrame]:
    """Read every data block of a STAR file.

    Parameters
    ----------
    path : str or path-like
        The STAR file.

    Returns
    -------
    dict of str to pandas.DataFrame
        One table per data block, keyed by block name (``""`` for an unnamed ``data_`` block). Column names
        have no leading underscore. Name columns (see :data:`NAME_COLUMNS`) are strings.
    """
    blocks = starfile.read(path, always_dict=True, parse_as_string=NAME_COLUMNS)
    return {name: block for name, block in blocks.items() if isinstance(block, pd.DataFrame)}


def read_star_table(path: str | PathLike, required: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    """Read the particle table of a STAR file.

    The table is the ``particles`` block if there is one, otherwise the only loop block in the file.

    Parameters
    ----------
    path : str or path-like
        The STAR file.
    required : list of str, optional
        Column names (without underscore) that must be present.

    Returns
    -------
    pandas.DataFrame
        The table.

    Raises
    ------
    ValueError
        If no single table can be chosen or a required column is missing.
    """
    blocks = read_star(path)
    if "particles" in blocks:
        table = blocks["particles"]
    elif len(blocks) == 1:
        table = next(iter(blocks.values()))
    else:
        raise ValueError(f"{path}: {len(blocks)} tables ({sorted(blocks)}) and none is named 'particles'")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; present: {list(table.columns)}")
    return table


def write_star(path: str | PathLike, table: pd.DataFrame | dict[str, pd.DataFrame], float_format: str = "%.6f") -> None:
    """Write one table (as an unnamed ``data_`` block) or several named blocks, tab-separated.

    Parameters
    ----------
    path : str or path-like
        Output file.
    table : pandas.DataFrame or dict of str to pandas.DataFrame
        The table, or blocks keyed by block name.
    float_format : str, optional
        printf-style format for floating-point columns.
    """
    starfile.write(table, path, float_format=float_format, sep="\t")
