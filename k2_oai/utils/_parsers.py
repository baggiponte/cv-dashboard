"""
Functions to parse various strings: file extensions, but also strings of coordinates
into Python lists.
"""

from __future__ import annotations

from ast import literal_eval

import numpy as np
from numpy.core.multiarray import ndarray


def _parse_extension(extension: str) -> str:
    """
    Makes sure an extension starts with '.' and is lowercase.

    Parameters
    ----------
    extension : str
        A string of a file extension.

    Returns
    -------
    str
        The same string with "." prepended (if missing) and lowercase.
    """
    return (
        f".{extension.lower()}" if not extension.startswith(".") else extension.lower()
    )


def parse_extensions(extensions: str | list[str]) -> list[str]:
    """
    Parses a string or a list of strings of file extension(s),
    and makes sure they are lowercase and start with a '.'. Then, returns a list.

    Parameters
    ----------
    extensions : str or list[str]
        A string or a list of strings of file extension(s).

    Returns
    -------
    list[str]
        A list of lowercase file extension(s). If missing, a '.' is prepended.
    """
    if isinstance(extensions, str):
        return [_parse_extension(extensions)]

    return [_parse_extension(extension) for extension in extensions]


def parse_str_as_array(
    array_string: str | ndarray,
    dtype: str = np.uint8,
    sort_coordinates: bool = False,
) -> ndarray:
    """Parses a string of coordinates into a Numpy `ndarray`.

    Parameters
    ----------
    array_string : str
        A string of coordinates.
    dtype: str or None (default = None)
        The datatype of the numpy array to be returned.
    sort_coordinates: bool (default = False)
        Whether to sort the coordinates.

    Returns
    -------
    ndarray
        A list of lists of integers, denoting pixel coordinates, i.e.
        [[x1, y1], [x2, y2], ...].
    """
    if isinstance(array_string, ndarray):
        return array_string
    elif isinstance(array_string, str):
        parsed_string: list[list[int]] = (
            sorted(literal_eval(array_string))
            if sort_coordinates
            else literal_eval(array_string)
        )
        return np.array(parsed_string, dtype=dtype)
    else:
        raise TypeError(
            f"Expected a string or a numpy array, but got {type(array_string)}"
        )
