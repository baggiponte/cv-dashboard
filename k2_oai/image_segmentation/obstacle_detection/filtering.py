import cv2 as cv
import numpy as np

from k2_oai.utils._args_checker import is_positive_odd_integer


def apply_filter(input_image: np.ndarray, sigma: int, method: str = "b"):
    """Applies a filter on the input image.

    Parameters
    ----------
    input_image : numpy.ndarray
        The image that the filter will be applied to.
    sigma : int
        The sigma value of the filter. It must be a positive, odd integer.
    method : str
        The method used to apply the filter. It must be either 'b' or 'g'.

    Returns
    -------
    numpy.ndarray
        The filtered image.
    """

    is_positive_odd_integer(sigma)

    if method == "b":
        return cv.bilateralFilter(input_image, 9, sigma, sigma)
    elif method == "g":
        return cv.GaussianBlur(input_image, (0, 0), sigma)
