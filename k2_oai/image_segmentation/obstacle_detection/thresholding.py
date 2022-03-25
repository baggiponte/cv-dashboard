import cv2 as cv
import numpy as np


def apply_binarization(
    input_image: np.ndarray, method: str = "s", C=0, blocksize=-1
) -> np.ndarray:
    """Applies a threshold on the grayscale input image. Depending on the method,
    applies simple thresholding (using the Otsu method to compute the threshold) or
    adaptive thresholding - which is more robust with respect to the different lighting
    conditions of the image.

    Parameters
    ----------
    input_image : numpy.ndarray
        The input image to which thresholding will be applied.
    method : str (default: "s")
        The thresholding method to use:
        - 's' stands for simple thresholding;
        - 'a' stands for adaptive thresholding
    C : int (default: 0)
        (Only for adaptive thresholding) The constant subtracted from the mean,
        or weighted mean. Normally is positive, but can also be negative.
    blocksize : int (default: -1)
        (Only for adaptive thresholding) Size of a pixel neighborhood, used
        to compute a threshold value for the pixel. Must be an odd number.

    Return
    ------
    numpy.ndarray
        The thresholded image.
    """

    if method == "s":
        retval, im_bin = cv.threshold(input_image, 0, 255, cv.THRESH_OTSU)

    elif method == "a":
        if blocksize > 0:
            if blocksize % 2 == 0:
                print("Insert a valid blocksize (odd number)")
                return
            else:
                kernel = blocksize
        else:
            kernel = int(input_image.size / 1000) + 1
            if kernel % 2 == 0:
                kernel = kernel + 1
        im_bin = cv.adaptiveThreshold(
            input_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel, C
        )

    else:
        print('Insert a valid method: "s" for simple, "a" for adaptive ')
        return

    if np.sum(im_bin == 255) > np.sum(im_bin == 0):
        im_bin = cv.bitwise_not(im_bin)

    return im_bin
