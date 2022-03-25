import cv2 as cv
import numpy as np


def compute_matrix(coordinates):
    # TODO: docstring
    """"""
    diff = np.subtract(coordinates[1], coordinates[0])
    theta = np.mod(np.arctan2(diff[0], diff[1]), np.pi / 2)
    center = coordinates[0]

    return cv.getRotationMatrix2D(
        (int(center[0]), int(center[1])), -theta * 180 / np.pi, 1
    )
