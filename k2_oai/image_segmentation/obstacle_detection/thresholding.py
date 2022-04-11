import cv2 as cv
import numpy as np


def otsu_algorithm(im_in, zeros):
  hist = cv.calcHist([im_in],[0],None,[256],[0,256])
  hist[0] = hist[0] - zeros #correction for images with masks
  hist_norm = hist.ravel()/hist.sum()
  Q = hist_norm.cumsum()
  bins = np.arange(256)
  fn_min = np.inf
  thresh = -1
  for i in range(1,256):
      p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
      q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
      if q1 < 1.e-6 or q2 < 1.e-6:
          continue
      b1,b2 = np.hsplit(bins,[i]) # weights
      # finding means and variances
      m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
      v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
      # calculates the minimization function
      fn = v1*q1 + v2*q2
      if fn < fn_min:
          fn_min = fn
          thresh = i

  return thresh, fn_min


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
    im_masked = cv.bitwise_and(input_image[:, :, 0], input_image[:, :, 3])
    n_zeros_mask = np.sum(input_image[:, :, 3] == 0)

    if method == "s":
        otsu_thresh, min_class_var = otsu_algorithm(im_masked, n_zeros_mask)
        retval, im_bin = cv.threshold(im_masked, otsu_thresh, 255, cv.THRESH_BINARY)

    elif method == "a":
        if blocksize > 0:
            if blocksize % 2 == 0:
                print("Insert a valid blocksize (odd number)")
                return
            else:
                kernel = blocksize
        else:
            kernel = int(im_masked.size / 1000) + 1
            if kernel % 2 == 0:
                kernel = kernel + 1
        im_bin = cv.adaptiveThreshold(
            im_masked, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel, C
        )

    elif method == 'c':
        otsu_thresh, min_class_var = otsu_algorithm(im_masked, n_zeros_mask)

        histSize = 256
        hist_gs = cv.calcHist([input_image],[0],None,[histSize],[0,256], accumulate=False)
        hist_gs[0] = hist_gs[0] - n_zeros_mask

        #tol = int(min_class_var/10)
        tol = int(np.var(hist_gs)/15000)
        print(tol)
        #tol = 30

        n_max = np.argmax(np.array(hist_gs))
        n_max = n_max*256/histSize

        retval, im_tresh_light = cv.threshold(input_image, n_max+tol, 255, cv.THRESH_BINARY)
        retval, im_tresh_dark = cv.threshold(input_image, n_max-tol, 255, cv.THRESH_BINARY_INV)
        im_bin = cv.bitwise_or(im_tresh_light, im_tresh_dark)
        im_bin = cv.bitwise_and(im_bin[:, :, 0], input_image[:, :, 3])

    else:
        print('Insert a valid method: "s" for simple, "a" for adaptive ')
        return

    if np.sum(im_bin == 255) > np.sum(im_bin == 0) - n_zeros_mask:
        im_bin = cv.bitwise_not(im_bin)
        im_bin = cv.bitwise_and(im_bin, input_image[:, :, 3])

    return im_bin
