from __future__ import annotations

import os

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from k2_oai.io import dropbox as dbx
from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)
from k2_oai.utils import draw_boundaries, rotate_and_crop_roof


@st.cache(allow_output_mutation=True)
def dbx_get_connection():
    if "DROPBOX_ACCESS_TOKEN" in os.environ:
        return dbx.dropbox_connect_access_token_only(st.session_state["access_token"])
    else:
        return dbx.dropbox_connect(
            st.session_state["access_token"], st.session_state["refresh_token"]
        )


@st.cache
def dbx_get_metadata(file_format: str = "parquet", dropbox_app=None):

    if file_format not in ["parquet", "csv"]:
        raise ValueError("file_format must be either 'parquet' or 'csv'")

    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app

    if file_format == "parquet":
        dbx_app.files_download_to_file(
            "join-roofs_images_obstacles.parquet",
            "/k2/metadata/transformed_data/join-roofs_images_obstacles.parquet",
        )
        metadata = pd.read_parquet("join-roofs_images_obstacles.parquet")

        if os.path.exists("join-roofs_images_obstacles.parquet"):
            os.remove("join-roofs_images_obstacles.parquet")

        return metadata

    dbx_app.files_download_to_file(
        "join-roofs_images_obstacles.csv",
        "/k2/metadata/raw_data/inner_join-roofs_images_obstacles.csv",
    )
    metadata = pd.read_csv("join-roofs_images_obstacles.csv")

    if os.path.exists("join-roofs_images_obstacles.csv"):
        os.remove("join-roofs_images_obstacles.csv")

    return metadata


@st.cache
def dbx_get_photos_list(folder_name, dropbox_app=None):
    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app
    return dbx.list_content_of(dbx_app, f"/k2/raw_photos/{folder_name}")


@st.cache
def dbx_get_photos_and_metadata(folder_name):

    dbx_app = dbx_get_connection()

    photos_list = dbx_get_photos_list(folder_name=folder_name, dropbox_app=dbx_app)

    photos_metadata = dbx_get_metadata(dropbox_app=dbx_app)

    available_photos_metadata = photos_metadata[
        photos_metadata.imageURL.isin(photos_list.item_name.values)
    ]

    return available_photos_metadata, photos_list


@st.cache(allow_output_mutation=True)
def dbx_load_photo(folder_name, photo_name, dropbox_app=None):
    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app
    dbx_app.files_download_to_file(
        photo_name, f"/k2/raw_photos/{folder_name}/{photo_name}"
    )

    bgr_image: np.ndarray = cv.imread(photo_name, 1)
    greyscale_image: np.ndarray = cv.imread(photo_name, 0)

    if os.path.exists(photo_name):
        os.remove(photo_name)

    return bgr_image, greyscale_image


def get_coordinates_from_roof_id(roof_id, photos_metadata):

    roof_px_coordinates = photos_metadata.loc[
        photos_metadata.roof_id == roof_id, "pixelCoordinates_roof"
    ].iloc[0]

    obstacles_px_coordinates = [
        coord
        for coord in photos_metadata.loc[
            photos_metadata.roof_id == roof_id, "pixelCoordinates_obstacle"
        ].values
    ]

    return roof_px_coordinates, obstacles_px_coordinates


def load_photos_from_roof_id(roof_id, photos_metadata, chosen_folder):
    photo_name = photos_metadata.loc[
        lambda df: df["roof_id"] == roof_id, "imageURL"
    ].values[0]

    return dbx_load_photo(chosen_folder, photo_name)


def crop_roofs_from_roof_id(roof_id, photos_metadata, chosen_folder):
    roof_px_coord, obstacles_px_coord = get_coordinates_from_roof_id(
        roof_id, photos_metadata
    )

    bgr_image, greyscale_image = load_photos_from_roof_id(
        roof_id, photos_metadata, chosen_folder
    )

    k2_labelled_image = draw_boundaries(bgr_image, roof_px_coord, obstacles_px_coord)
    bgr_roof = rotate_and_crop_roof(k2_labelled_image, roof_px_coord)
    greyscale_roof = rotate_and_crop_roof(greyscale_image, roof_px_coord)

    return k2_labelled_image, bgr_roof, greyscale_roof


def obstacle_detection_pipeline(
    greyscale_roof,
    sigma,
    filtering_method,
    binarization_method,
    blocksize,
    tolerance,
    boundary_type,
    return_filtered_roof: bool = False,
):

    filtered_roof = filtering_step(greyscale_roof, sigma, filtering_method.lower())

    if binarization_method == "Simple":
        binarized_roof = binarization_step(filtered_roof, method="s")
    elif binarization_method == "Adaptive":
        binarized_roof = binarization_step(
            filtered_roof, method="a", adaptive_kernel_size=blocksize
        )
    else:
        binarized_roof = binarization_step(
            filtered_roof, method="c", composite_tolerance=int(tolerance)
        )

    blurred_roof = morphological_opening_step(binarized_roof)

    boundary_type = "box" if boundary_type == "Bounding Box" else "polygon"

    blobs, roof_with_bboxes, obstacles_coordinates = detect_obstacles(
        blurred_roof=blurred_roof,
        source_image=greyscale_roof,
        box_or_polygon=boundary_type,
        min_area="auto",
    )

    if return_filtered_roof:
        return blobs, roof_with_bboxes, obstacles_coordinates, filtered_roof
    return blobs, roof_with_bboxes, obstacles_coordinates
