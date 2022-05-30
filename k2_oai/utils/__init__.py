"""
Various utilities for the K2 OAI project. See the README.md for more information.
"""

from ._args_checker import is_positive_odd_integer, is_valid_method
from ._image_manipulation import (
    draw_labels,
    pad_image,
    read_image_from_bytestring,
    rotate_and_crop_roof,
)
from ._parsers import parse_str_as_coordinates
