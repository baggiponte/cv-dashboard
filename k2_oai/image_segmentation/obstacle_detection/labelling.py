from typing import Any

import cv2 as cv
import numpy as np


def bounding_boxes(draw_image, stats, min_area, margin_w, margin_h):

    rect_coord = []

    for i in range(0, stats.shape[0]):

        #if stats[i, cv.CC_STAT_AREA] > min_area:

        topleft_p = (
            stats[i, cv.CC_STAT_LEFT] + margin_w,
            stats[i, cv.CC_STAT_TOP] + margin_h,
        )
        h = stats[i, cv.CC_STAT_HEIGHT]
        w = stats[i, cv.CC_STAT_WIDTH]
        botright_p = (topleft_p[0] + w, topleft_p[1] + h)

        #if h < draw_image.shape[0]*0.8 and w < draw_image.shape[1]*0.8:

        rect_coord.append((topleft_p, botright_p))
        draw_image = cv.rectangle(draw_image, topleft_p, botright_p, (255, 0, 0), 1)

    return rect_coord, draw_image


def oriented_bboxes(im_labeled, draw_image, stats, min_area, margin_w, margin_h):
    rect_coord = []

    for i in range(0, stats.shape[0]):

        #if stats[i, cv.CC_STAT_AREA] > min_area:

        obst_im = (im_labeled == i)*255
        contours, hierarchy = cv.findContours(obst_im.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        approxCurve = cv.approxPolyDP(contours[0], 5.0, True)
        cv.polylines(draw_image, [approxCurve], True, (255, 0, 0), 2)
        rect_coord.append(approxCurve)

    return rect_coord, draw_image


def image_segmentation(
    input_image: np.ndarray,
    labelled_image: np.ndarray,
    kernel_opening: int,
    min_area: int = 0,
    cut_border: int = 0,
    bbox_or_polygon: str = "bbox"
):
    """Finds the connected components in a binary image and assigns a label to them.
    First, crops the border of the image (depending on the cut_border parameter), then
    applies a morphological opening. After the connected components' analysis, the
    algorithm rejects the components having an area less than the area_min parameter.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image.
    labelled_image : numpy.ndarray
        The image where obstacles have been labelled.
    kernel_opening : int
        Size of the kernel used for the morphological opening.
        Must be a positive, odd number.
    min_area : int (default=10)
        Minimum area of the connected components to be kept.
    cut_border : int (default=0)
        Fraction of the image shape that has to be cut from the borders.
    bbox_or_polygon : str (default='bbox')
        String indicating whether to using bounding boxes or bounding polygon.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[tuple[str]]]
        Returns a tuple of three objects:
        - The cropped image with labels
        - The image with the bounding box of the labels,
        - The list of coordinates of the top-left and bottom-right points of the
          bounding boxes of the obstacles that have been found.
    """

    # cut border
    if cut_border > 0:
        margin_h = int(input_image.shape[0] / cut_border)
        margin_w = int(input_image.shape[1] / cut_border)
    else:
        margin_h = 0
        margin_w = 0

    im_cut_border = input_image[
        margin_h : input_image.shape[0] - margin_h,
        margin_w : input_image.shape[1] - margin_w,
    ]

    kernel = np.ones((kernel_opening, kernel_opening), np.uint8)
    im_open = cv.morphologyEx(im_cut_border, cv.MORPH_OPEN, kernel)
    im_close = cv.morphologyEx(im_open, cv.MORPH_CLOSE, kernel)
    num_labels, im_labeled, stats, centroids = cv.connectedComponentsWithStats(
        im_close, connectivity=8
    )

    draw_image = cv.cvtColor(labelled_image, cv.COLOR_BGRA2BGR)

    if bbox_or_polygon == "bbox":
        bbox_coord, bbox_image = bounding_boxes(draw_image, stats, min_area, margin_w, margin_h)
    elif bbox_or_polygon == "polygon":
        bbox_coord, bbox_image = oriented_bboxes(im_labeled, draw_image, stats, min_area, margin_w, margin_h)
    else:
        bbox_coord, bbox_image = bounding_boxes(draw_image, stats, min_area, margin_w, margin_h)

    return im_labeled, bbox_image, bbox_coord
