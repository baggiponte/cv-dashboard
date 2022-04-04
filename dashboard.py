import ast
import os
import time

import cv2 as cv
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
from PIL import Image
import streamlit.components.v1 as components

from k2_oai.image_segmentation import *
from k2_oai.utils._draw_boundaries import *
from k2_oai.metrics import surface_absolute_error

from k2_oai.utils.dropbox_io_utils import *

st.set_page_config(
    page_title="K2 <-> OAI",
    layout='wide',
    initial_sidebar_state='auto'
)

st.title("Obstacle detection dashboard")


@st.cache
def get_dropbox_data_structures():
    _dbx = dropbox_connect()
    st_dropbox_list_files_df = get_dropbox_list_files_df(_dbx, "/k2/raw_photos/small_photos-api_upload")
    _dbx.files_download_to_file(
        "inner_join-roofs_images_obstacles.csv",
        "/k2/metadata/raw_data/inner_join-roofs_images_obstacles.csv"
    )
    metadata_df = pd.read_csv("inner_join-roofs_images_obstacles.csv")
    return st_dropbox_list_files_df, metadata_df


@st.cache(allow_output_mutation=True)
def load_photo_from_dropbox(photo_name):
    _dbx = dropbox_connect()
    _dbx.files_download_to_file(
       photo_name,
       "/k2/raw_photos/small_photos-api_upload/{}".format(photo_name)
    )
    im_bgr: np.ndarray = cv.imread(photo_name, 1)
    im_gs: np.ndarray = cv.imread(photo_name, 0)
    return im_bgr, im_gs


def plot_channel_histogram(im_in):
    return cv.calcHist(im_in, [0], None, [256], [0, 256])


dropbbox_list_files_df, metadata_df = get_dropbox_data_structures()

photos_list = dropbbox_list_files_df.item_abs_path.values
if os.path.exists("inner_join-roofs_images_obstacles.csv"):
    os.remove("inner_join-roofs_images_obstacles.csv")

st.write(dropbbox_list_files_df.shape)
with st.expander("Click to see the list of photos paths on remote folder"):
    st.dataframe(dropbbox_list_files_df)

metadata_df = metadata_df[metadata_df.imageURL.isin(dropbbox_list_files_df.item_name.values)]
with st.expander("Click to see the metadata table for available photos"):
    st.dataframe(metadata_df)

roof_id = None
try:
    roof_id = int(st.selectbox("Select roof_id: ", options=sorted(metadata_df.roof_id.unique())))
except:
    st.error("Please insert a valid roof_id")

st.write("Roof obstacles metadata as recorded in DB:")

metadata_photo_obstacles = metadata_df.loc[
    metadata_df.roof_id == roof_id,
]

st.dataframe(metadata_photo_obstacles)

pixel_coord_roof = metadata_df.loc[
    metadata_df.roof_id == roof_id,
    "pixelCoordinates_roof"
].iloc[0]

pixel_coord_obs = [
    coord for coord in metadata_df.loc[
        metadata_df.roof_id == roof_id,
        "pixelCoordinates_obstacle"
    ].values
]

st.write("Number of obstacles in DB:", len(pixel_coord_obs))

test_photo_name = metadata_df.loc[lambda df: df["roof_id"] == roof_id, "imageURL"].values[0]

try:

    im_bgr, im_gs = load_photo_from_dropbox(test_photo_name)
    os.remove(test_photo_name)

    im_k2labeled = draw_boundaries(
        im_bgr, pixel_coord_roof, pixel_coord_obs
    )

    im_gs_cropped = rotate_and_crop_roof(im_gs, pixel_coord_roof)
    im_bgr_cropped = rotate_and_crop_roof(im_k2labeled, pixel_coord_roof)

    im_filtered = apply_filter(
        im_gs_cropped,
        301,
        "b"
    )

    im_thresholded = apply_binarization(im_filtered, "s")

    if im_gs_cropped.size > 10000:
        opening = 3
        edges = 12
    else:
        opening = 1
        edges = 15

    im_segmented, im_bbox, rect_coord = image_segmentation(
        im_thresholded,
        im_gs_cropped,
        opening,
        min_area=int(np.max(im_thresholded.shape)/10),
        cut_border=edges
    )

    im_draw, im_result, im_error, im_rel_area_error = surface_absolute_error(
        im_thresholded,
        pixel_coord_roof,
        pixel_coord_obs,
        rect_coord
    )

    st_cols = st.columns((1, 1))
    st_cols[0].image(im_k2labeled, use_column_width=True, channels="BGR", caption="Original image with DB labels")
    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 0].flatten(), bins=50, edgecolor='blue', alpha=0.5)
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 1].flatten(), bins=50, edgecolor='green', alpha=0.5)
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 2].flatten(), bins=50, edgecolor='red', alpha=0.5)
    ax.set_title("RGB crop histogram")
    ax.set_xlim(0, 255)
    st_cols[1].pyplot(fig, use_column_width=True)

    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(im_gs_cropped.flatten(), bins=range(256), edgecolor='black', alpha=0.9)
    ax.set_title("GS crop histogram")
    ax.set_xlim(0, 255)
    st_cols[1].pyplot(fig, use_column_width=True)

    st_cols = st.columns((1, 1, 1))

    st_cols[0].image(im_bgr_cropped, use_column_width=True, channels="BGR", caption="Cropped Roof RGB with DB labels")
    st_cols[1].image(im_gs_cropped, use_column_width=True, caption="Cropped Roof GS")
    st_cols[2].image(im_filtered, use_column_width=True, caption="Cropped Roof GS filtered")

    st_cols = st.columns((1, 1, 1))

    st_cols[0].image((im_segmented * 60) % 256, use_column_width=True, caption="Auto GS Blobs")
    st_cols[1].image(im_bbox, use_column_width=True, caption="Auto labels BBOX")
    st_cols[2].image(im_error, use_column_width=True, caption="Auto labels VS DB labels")

except:

    st.error("Photo corresponding to roof_id={} not found in the remote folder!".format(roof_id))
