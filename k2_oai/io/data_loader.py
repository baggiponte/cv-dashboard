"""
Loads and manipulates data from dropbox
"""

import os

import cv2 as cv
import pandas as pd

from k2_oai.utils import draw_boundaries, rotate_and_crop_roof

__all__ = (
    "dbx_load_dataframe",
    "dbx_get_metadata",
    "dbx_load_photo",
    "crop_roofs_from_roof_id",
)


def dbx_load_dataframe(filename, dropbox_path, dropbox_app=None):

    dropbox_file = f"{dropbox_path}/{filename}"

    dropbox_app.files_download_to_file(filename, dropbox_file)

    if filename.endswith(".parquet"):
        data = pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        data = pd.read_csv(filename)
    else:
        raise ValueError("File must be either .parquet or .csv")

    if os.path.exists(filename):
        os.remove(filename)

    return data


def dbx_get_metadata(file_format: str = "parquet", dropbox_path=None, dropbox_app=None):
    if file_format not in ["parquet", "csv"]:
        raise ValueError("file_format must be either 'parquet' or 'csv'")

    path = dropbox_path or "/k2/metadata/transformed_data"

    return dbx_load_dataframe(
        f"join-roofs_images_obstacles.{file_format}",
        dropbox_path=path,
        dropbox_app=dropbox_app,
    )


# TODO: convert code in notebook into functions here
# def dbx_get_labels_quality_data(dropbox_app):


def dbx_load_photo(photo_name, dropbox_path, dropbox_app, greyscale_only: bool = False):

    download_path = f"{photo_name}"
    dropbox_path = f"{dropbox_path}/{photo_name}"

    dropbox_app.files_download_to_file(download_path, dropbox_path)

    if greyscale_only:
        greyscale_image = cv.imread(photo_name, 0)

        if os.path.exists(download_path):
            os.remove(download_path)

        return greyscale_image

    bgr_image = cv.imread(photo_name, 1)
    greyscale_image = cv.imread(photo_name, 0)

    if os.path.exists(download_path):
        os.remove(download_path)

    return bgr_image, greyscale_image


def _get_coordinates_from_roof_id(roof_id, photos_metadata) -> tuple[str, list[str]]:

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


def _load_photos_from_roof_id(
    roof_id,
    photos_metadata,
    dropbox_path,
    dropbox_app,
    greyscale_only: bool = False,
):
    photo_name = photos_metadata.loc[
        lambda df: df["roof_id"] == roof_id, "imageURL"
    ].values[0]

    return dbx_load_photo(photo_name, dropbox_path, dropbox_app, greyscale_only)


def crop_roofs_from_roof_id(
    roof_id,
    photos_metadata,
    dropbox_path,
    dropbox_app,
    greyscale_only: bool = False,
):
    roof_px_coord, obstacles_px_coord = _get_coordinates_from_roof_id(
        roof_id, photos_metadata
    )

    if greyscale_only:
        greyscale_image = _load_photos_from_roof_id(
            roof_id, photos_metadata, dropbox_path, dropbox_app, greyscale_only
        )
        return rotate_and_crop_roof(greyscale_image, roof_px_coord)

    bgr_image, greyscale_image = _load_photos_from_roof_id(
        roof_id, photos_metadata, dropbox_path, dropbox_app
    )

    k2_labelled_image = draw_boundaries(bgr_image, roof_px_coord, obstacles_px_coord)
    bgr_roof = rotate_and_crop_roof(k2_labelled_image, roof_px_coord)
    greyscale_roof = rotate_and_crop_roof(greyscale_image, roof_px_coord)

    return k2_labelled_image, bgr_roof, greyscale_roof
