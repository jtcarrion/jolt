"""Read and write IMOD files: positional ``.xf`` alignments, ``.tlt`` tilt angles and ``.mod`` models.

An ``.xf`` row applies to the stack image at the same position; rows are never sorted by tilt angle.
A ``.tlt`` whose length disagrees with its ``.xf`` is an error, never padded or truncated.
"""
