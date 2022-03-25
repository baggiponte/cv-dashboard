from __future__ import annotations

from ast import literal_eval

import numpy as np


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


def parse_str_as_coordinates(
        string: str, dtype: str | None = None,
        sort_coords: bool = False
) -> np.array:
    """Parses a string of coordinates into a list of lists of strings.

    Parameters
    ----------
    string : str
        A string of coordinates or a list of lists of coordinates.
    dtype: str or None (default = None)
        The datatype of the numpy array to be returned.

    Returns
    -------
    np.ndarray
        A list of lists of integers, denoting pixel coordinates, i.e.
        [[x1, y1], [x2, y2], ...].
    """
    if not isinstance(string, str):
        raise TypeError(f"{type(string)} is not a valid type.")
    if sort_coords:
        parsed_string = sorted(literal_eval(string))
    else:
        parsed_string = literal_eval(string)
    if dtype:
        return np.array(parsed_string, dtype=dtype)
    return np.array(parsed_string)


# def parse_coordinates_as_lists(
#     data: DataFrame, cols: list[str] | str | None = None
# ) -> DataFrame:
#     if cols is None:
#         target_cols: list[str] = [
#             col for col in data.columns if col.startswith("pixelCoordinates")
#         ]
#     else:
#         target_cols: list[str] = [col for col in data.columns if col in cols]
#
#     df = data.copy()
#     for col in target_cols:
#         df[col] = data[col].apply(lambda x: literal_eval(x))
#
#     return df
