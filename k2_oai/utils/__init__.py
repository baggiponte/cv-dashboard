"""
Various utilities for the K2 OAI project. See the README.md for more information.
"""

from ._args_checker import is_positive_odd_integer, is_valid_method
from ._image_manipulation import (
    _compute_rotation_matrix,
    draw_boundaries,
    pad_image,
    read_image_from_bytestring,
    rotate_and_crop_roof,
)
from ._parsers import experimental_parse_str_as_array, parse_str_as_coordinates
