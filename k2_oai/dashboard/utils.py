from __future__ import annotations

import os

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st

from k2_oai.io import dropbox as dbx


@st.cache(allow_output_mutation=True)
def get_dropbox_connection():
    if "DROPBOX_ACCESS_TOKEN" in os.environ:
        return dbx.dropbox_connect_access_token_only(st.session_state["access_token"])
    else:
        return dbx.dropbox_connect(
            st.session_state["access_token"], st.session_state["refresh_token"]
        )


@st.cache
def get_photos_metadata(file_format: str = "parquet", dropbox_app=None):

    if file_format not in ["parquet", "csv"]:
        raise ValueError("file_format must be either 'parquet' or 'csv'")

    dbx_app = get_dropbox_connection() if dropbox_app is None else dropbox_app

    if file_format == "parquet":
        dbx_app.files_download_to_file(
            "join-roofs_images_obstacles.parquet",
            "/k2/metadata/transformed_data/join-roofs_images_obstacles.parquet",
        )
        return pd.read_parquet("join-roofs_images_obstacles.parquet")

    dbx_app.files_download_to_file(
        "join-roofs_images_obstacles.csv",
        "/k2/metadata/transformed_data/join-roofs_images_obstacles.csv",
    )
    return pd.read_csv("join-roofs_images_obstacles.csv")


@st.cache
def get_photos_list(folder_name, dropbox_app=None):
    dbx_app = get_dropbox_connection() if dropbox_app is None else dropbox_app
    return dbx.list_content_of(dbx_app, f"/k2/raw_photos/{folder_name}")


@st.cache
def get_photos_and_metadata_from_dbx(folder_name):

    dbx_app = get_dropbox_connection()
    photos_list = get_photos_list(folder_name=folder_name, dropbox_app=dbx_app)
    photos_metadata = get_photos_metadata(dropbox_app=dbx_app)

    return photos_list, photos_metadata


@st.cache(allow_output_mutation=True)
def load_photo_from_dbx(folder_name, photo_name):
    dbx_app = get_dropbox_connection()
    dbx_app.files_download_to_file(
        photo_name, f"/k2/raw_photos/{folder_name}/{photo_name}"
    )
    im_bgr: np.ndarray = cv.imread(photo_name, 1)
    im_gs: np.ndarray = cv.imread(photo_name, 0)
    return im_bgr, im_gs
