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

    if method == 'b':

        if len(input_image.shape) > 2:
            if input_image.shape[2] > 3:#bgra image
                im_use = cv.cvtColor(input_image, cv.COLOR_BGRA2BGR)
                im_result = input_image
                im_result[:, :, 0:3] = cv.bilateralFilter(im_use, 9, sigma, sigma)
            else:#bgr image
                im_result = cv.bilateralFilter(input_image, 9, sigma, sigma)
                im_result = cv.cvtColor(im_result, cv.COLOR_BGR2BGRA)
        else:#grayscale image
            im_result = cv.bilateralFilter(input_image, 9, sigma, sigma)
            im_result = cv.cvtColor(im_result, cv.COLOR_GRAY2BGRA)

        return im_result

    elif method == 'g':
        return cv.GaussianBlur(input_image, (0, 0), sigma)

    else:
        print("Insert 'b' for bilater filter or 'g' for Gaussian blur")
