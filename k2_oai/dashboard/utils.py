from __future__ import annotations

import os

import cv2 as cv
import dropbox
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from k2_oai.io import dropbox as dbx
from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)
from k2_oai.utils import draw_boundaries, rotate_and_crop_roof

load_dotenv()

DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
DROPBOX_USER_EMAIL = os.environ.get("DROPBOX_USER_MAIL")
DROPBOX_APP_KEY = os.environ.get("APP_KEY")
DROPBOX_APP_SECRET = os.environ.get("APP_SECRET")
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")


def st_dropbox_oauth2_connect(
    dbx_app_key=DROPBOX_APP_KEY, dbx_app_secret=DROPBOX_APP_SECRET
):
    dropbox_oauth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dbx_app_key,
        dbx_app_secret,
        token_access_type="offline",
    )

    authorization_url = dropbox_oauth_flow.start()

    placeholder = st.empty()

    with placeholder.container():
        st.title(":key: Dropbox Authentication")
        st.markdown(f"1. Go to [this url]({authorization_url}).")
        st.write('2. Click "Allow" (you might have to log in first).')
        st.write("3. Copy the authorization code.")
        st.write("4. Enter the authorization code here: ")
        authorization_code = st.text_input("")

    try:
        complete_oauth_flow = dropbox_oauth_flow.finish(authorization_code)
        return placeholder, complete_oauth_flow
    except:  # noqa: E722
        if authorization_code is not None and len(authorization_code):
            st.error("Invalid authorization code!")
        return placeholder, None


@st.cache(allow_output_mutation=True)
def dbx_get_connection():
    if "DROPBOX_ACCESS_TOKEN" in os.environ:
        return dbx.dropbox_connect_access_token_only(st.session_state["access_token"])
    else:
        return dbx.dropbox_connect(
            st.session_state["access_token"], st.session_state["refresh_token"]
        )


@st.cache
def dbx_list_dir_contents(folder_name, dropbox_app=None):
    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app
    return dbx.list_content_of(dbx_app, folder_name)


@st.cache
def dbx_load_dataframe(filename, root_path, dropbox_app=None):
    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app

    dbx_app.files_download_to_file(f"/tmp/{filename}", f"{root_path}/{filename}")

    if filename.endswith(".parquet"):
        data = pd.read_parquet(f"/tmp/{filename}")
    elif filename.endswith(".csv"):
        data = pd.read_csv(f"/tmp/{filename}")
    else:
        raise ValueError("File must be either .parquet or .csv")

    if os.path.exists(f"/tmp/{filename}"):
        os.remove(f"/tmp/{filename}")

    return data


@st.cache
def dbx_get_metadata(file_format: str = "parquet", dropbox_app=None):
    if file_format not in ["parquet", "csv"]:
        raise ValueError("file_format must be either 'parquet' or 'csv'")

    dbx_app = dbx_get_connection() if dropbox_app is None else dropbox_app

    return dbx_load_dataframe(
        f"join-roofs_images_obstacles.{file_format}",
        root_path="/k2/metadata/transformed_data",
        dropbox_app=dbx_app,
    )


@st.cache
def dbx_get_photo_list_and_metadata(photos_folder, photos_root_path):

    full_path = f"{photos_root_path}/{photos_folder}"

    dbx_app = dbx_get_connection()

    photos_list = dbx_list_dir_contents(folder_name=full_path, dropbox_app=dbx_app)

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

    bgr_image = cv.imread(photo_name, 1)
    greyscale_image = cv.imread(photo_name, 0)

    if os.path.exists(photo_name):
        os.remove(photo_name)

    return bgr_image, greyscale_image


def get_coordinates_from_roof_id(roof_id, photos_metadata) -> tuple[str, list[str]]:

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
