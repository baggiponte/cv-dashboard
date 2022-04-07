from __future__ import annotations

import cv2 as cv
import numpy as np

from ._linalg import compute_matrix
from ._parsers import parse_str_as_coordinates


def draw_boundaries(
    input_image: np.ndarray,
    roof_coordinates: str,
    obstacle_coordinates: str,
) -> np.ndarray:
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    input_image : np.ndarray
        Input image.
    roof_coordinates : str or list of strings
        Roof coordinates.
    obstacle_coordinates : str
        Obstacle coordinates.

    Returns
    -------
    np.ndarray
        Image with labels drawn.
    """
    points: np.array = parse_str_as_coordinates(roof_coordinates).reshape((-1, 1, 2))
    result: np.ndarray = cv.polylines(input_image, [points], True, (0, 0, 255), 2)

    for obst in obstacle_coordinates:
        points: np.array = parse_str_as_coordinates(obst).reshape((-1, 1, 2))
        result: np.array = cv.polylines(result, [points], True, (255, 0, 0), 2)

    return result


def rotate_and_crop_roof(input_image: np.ndarray, roof_coordinates: str) -> np.ndarray:
    """Rotates the input image to make the roof sides parallel to the image,
    then crops it.

    Parameters
    ----------
    input_image : np.ndarray
        The input image.
    roof_coordinates : str
        Roof coordinates.

    Returns
    -------
    np.ndarray
        The rotated and cropped roof.
    """

    coord = parse_str_as_coordinates(roof_coordinates, dtype="int32", sort_coords=True)
    
    if len(input_image.shape) < 3:
        im_alpha = cv.cvtColor(input_image, cv.COLOR_GRAY2BGRA)
    else:
        im_alpha = cv.cvtColor(input_image, cv.COLOR_BGR2BGRA)

    #rectangular roofs
    if len(coord) == 4:
        M = compute_matrix(coord)

        im_affine = cv.warpAffine(
            im_alpha, M, im_alpha.shape[0:2], cv.INTER_LINEAR, cv.BORDER_CONSTANT
        )

        diff = np.subtract(coord[1], coord[0])

        if diff[1] > 0:
            dist_y = np.linalg.norm(coord[1] - coord[0]).astype(int)
            dist_x = np.linalg.norm(coord[2] - coord[0]).astype(int)
        else:
            dist_y = np.linalg.norm(coord[2] - coord[0]).astype(int)
            dist_x = np.linalg.norm(coord[1] - coord[0]).astype(int)

        im_result = im_affine[coord[0][1] : coord[0][1] + dist_y, coord[0][0] : coord[0][0] + dist_x, :]

    #polygonal roofs
    else:
        mask = np.zeros(input_image.shape[0:2], dtype="uint8")

        pts = np.array(coord, np.int32)  
        pts = pts.reshape((-1,1,2))

        cv.fillConvexPoly(mask, pts, (255, 255, 255))

        im_alpha[:, :, 3] = mask

        bot_right = np.max(pts, axis=0)
        top_left = np.min(pts, axis=0)

        im_result = im_alpha[top_left[0][1] : bot_right[0][1], top_left[0][0] : bot_right[0][0], :]

    return im_result
