"""
Functions to parse various strings: file extensions, but also strings of coordinates
into Python lists.
"""

from __future__ import annotations

from ast import literal_eval

import numpy as np
from numpy.core.multiarray import ndarray


def parse_str_as_coordinates(
    string: str, dtype=np.int32, sort_coordinates: bool = False
) -> np.array:
    """Parses a string of coordinates into a list of lists of strings.
    Parameters
    ----------
    string : str
        A string of coordinates.
    dtype: numpy dtype (default = np.int32)
        The datatype of the numpy array to be returned.
    sort_coordinates: bool (default = False)
        Whether to sort the coordinates.
    Returns
    -------
    np.ndarray
        A list of lists of integers, denoting pixel coordinates, i.e.
        [[x1, y1], [x2, y2], ...].
    """
    if not isinstance(string, str):
        raise TypeError(f"{type(string)} is not a valid type.")

    parsed_string = (
        sorted(literal_eval(string)) if sort_coordinates else literal_eval(string)
    )

    if dtype:
        return np.array(parsed_string, dtype=dtype)
    return np.array(parsed_string)


def experimental_parse_str_as_array(
    array_string: str | ndarray,
    sort_coordinates: bool = False,
    dtype=np.uint8,
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
