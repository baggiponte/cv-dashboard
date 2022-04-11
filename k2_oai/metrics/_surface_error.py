from ast import literal_eval
from typing import Any

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from k2_oai.utils._linalg import compute_matrix
from k2_oai.utils._parsers import parse_str_as_coordinates


def surface_absolute_error(
    input_image: np.array, roof_coordinates, obstacle_coordinates, label_coord
):
    # TODO: add docstrings

    obstacle_number: list[Any] = [literal_eval(obst) for obst in obstacle_coordinates]
    coord: np.array = parse_str_as_coordinates(roof_coordinates, dtype="int32", sort_coords=True)
    M = compute_matrix(coord)

    black = np.zeros(input_image.shape, np.uint8)

    for obst in obstacle_number:
        pts = []
        for obst_coord in obst:
            obst_coord.append(1)
            coord_ext = np.array(obst_coord)
            pts_rot = np.matmul(M, coord_ext)

            pts.append(pts_rot.tolist())

        pts_ar = np.subtract(np.array(pts), np.array(coord[0])).astype(int)
        im_draw = cv.fillConvexPoly(black, pts_ar, (255, 255, 255), 1)

    # labeled by us image
    im_result = np.zeros(input_image.shape, np.uint8)
    for label in label_coord:
        im_result = cv.fillConvexPoly(im_result, label, (255, 255, 255), 1)

    im_error = cv.bitwise_xor(im_result, im_draw)

    return im_draw, im_result, im_error, np.sum(im_error == 255) / im_error.size * 100
