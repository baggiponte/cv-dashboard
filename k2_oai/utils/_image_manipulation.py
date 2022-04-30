"""
This module contains two auxiliary functions for the obstacle detection module.
For example, draws the boundaries of roofs and obstacles on a given image,
rotates and crops the roofs, or applies padding to an image.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy.core.multiarray import ndarray

from k2_oai.utils._parsers import parse_str_as_array


def read_image_from_bytestring(
    bytestring_image: bytes,
    as_greyscale: bool = True,
) -> ndarray:
    """Reads the bytestring and returns it as a numpy array.
    This passage is necessary because the API sends a file that is transferred
    to the server as a bytestring.

    Parameters
    ----------
    bytestring_image : bytes
        The bytestring of the image.
    as_greyscale : bool
        If True, the image is converted to greyscale. The default is True.

    Returns
    -------
    ndarray
        The image as a numpy array.
    """
    image_array: ndarray = np.fromstring(bytestring_image, np.uint8)

    if as_greyscale:
        return cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)
    return cv.imdecode(image_array, cv.IMREAD_COLOR)


def pad_image(
    image: ndarray,
    padding_percentage: int | None = None,
) -> tuple[ndarray, tuple[int, int]]:
    """Applies padding to an image (e.g. to remove borders).

    Parameters
    ----------
    image : ndarray
        The image to be padded.
    padding_percentage : int or None (default: None)
        The size of the padding, as integer between 1 and 100.
        If None, no padding is applied.

    Returns
    -------
    ndarray, tuple[int, int]
        The padded image and the margins for the padding.
    """
    if padding_percentage not in range(0, 101):
        raise ValueError("Parameter `padding` must range between 1 and 100.")
    elif padding_percentage is None or padding_percentage == 0:
        margin_h, margin_w = 0, 0
    else:
        margin_h, margin_w = (
            int(image.shape[n] / padding_percentage) for n in range(2)
        )

    padded_image: ndarray = image[
        margin_h : image.shape[0] - margin_h,
        margin_w : image.shape[1] - margin_w,
    ]

    return padded_image, (margin_h, margin_w)


def draw_boundaries(
    input_image: ndarray,
    roof_coordinates: str | ndarray,
    obstacle_coordinates: str | ndarray | list[str] | list[ndarray] | None = None,
) -> ndarray:
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    input_image : ndarray
        Input image.
    roof_coordinates : str or ndarray
        Roof coordinates, either as string or list of lists of integers.
    obstacle_coordinates : str or ndarray or None (default: None)
        Obstacle coordinates. Can be None if there are no obstacles. Defaults to None.

    Returns
    -------
    ndarray
        Image with labels drawn.
    """

    if isinstance(roof_coordinates, str):
        roof_coordinates: ndarray = parse_str_as_array(
            roof_coordinates, sort_coordinates=True
        )

    obstacle_coordinate_pairs: ndarray = roof_coordinates.reshape((-1, 1, 2))
    result: ndarray = cv.polylines(
        input_image, [obstacle_coordinate_pairs], True, (0, 0, 255), 2
    )

    if obstacle_coordinates:
        if isinstance(obstacle_coordinates, str):
            obstacle_coordinates: ndarray = parse_str_as_array(
                obstacle_coordinates, sort_coordinates=True
            )
        for obstacle in obstacle_coordinates:
            obstacle_coordinate_pairs: ndarray = obstacle.reshape((-1, 1, 2))
            result: ndarray = cv.polylines(
                result, [obstacle_coordinate_pairs], True, (255, 0, 0), 2
            )

    return result


def compute_rotation_matrix(coordinates_array):
    diff = np.subtract(coordinates_array[1], coordinates_array[0])
    theta = np.mod(np.arctan2(diff[0], diff[1]), np.pi / 2)
    center = coordinates_array[0]
    rotation_matrix: ndarray = cv.getRotationMatrix2D(
        (int(center[0]), int(center[1])), -theta * 180 / np.pi, 1
    )
    return diff, rotation_matrix


def rotate_and_crop_roof(input_image: ndarray, roof_coordinates: str) -> ndarray:
    """Rotates the input image to make the roof sides parallel to the image,
    then crops it.

    Parameters
    ----------
    input_image : ndarray
        The input image.
    roof_coordinates : str
        Roof coordinates: if string, it is parsed as a string of coordinates
        (i.e. a list of list of integers: [[x1, y1], [x2, y2], ...]).

    Returns
    -------
    ndarray
        The rotated and cropped roof.
    """
    coordinates_array: ndarray = parse_str_as_array(
        roof_coordinates, sort_coordinates=True
    )

    diff, rotation_matrix = compute_rotation_matrix(coordinates_array)

    rotated_image: ndarray = cv.warpAffine(
        input_image,
        rotation_matrix,
        input_image.shape[0:2],
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
    )

    if diff[1] > 0:
        dist_y = np.linalg.norm(coordinates_array[1] - coordinates_array[0]).astype(int)
        dist_x = np.linalg.norm(coordinates_array[2] - coordinates_array[0]).astype(int)
    else:
        dist_y = np.linalg.norm(coordinates_array[2] - coordinates_array[0]).astype(int)
        dist_x = np.linalg.norm(coordinates_array[1] - coordinates_array[0]).astype(int)

    return rotated_image[
        coordinates_array[0][1] : coordinates_array[0][1] + dist_y,
        coordinates_array[0][0] : coordinates_array[0][0] + dist_x,
    ]
