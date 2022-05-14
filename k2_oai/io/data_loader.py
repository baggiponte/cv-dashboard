"""
Loads data and photos from Dropbox
"""

import os

import cv2 as cv
import geopandas
import pandas as pd

from k2_oai.io import dropbox as dbx
from k2_oai.io.dropbox_paths import (
    DROPBOX_ANNOTATIONS_PATH,
    DROPBOX_EXTERNAL_DATA_PATH,
    DROPBOX_METADATA_PATH,
)
from k2_oai.utils import draw_boundaries, rotate_and_crop_roof

__all__ = [
    "dbx_load_dataframe",
    "dbx_load_metadata",
    "dbx_load_geo_metadata",
    "dbx_load_earth",
    "dbx_load_label_annotations",
    "dbx_load_photo",
    "dbx_load_photos_from_roof_id",
]


def dbx_load_dataframe(filename, dropbox_path, dropbox_app):

    dropbox_file = f"{dropbox_path}/{filename}"

    dropbox_app.files_download_to_file(filename, dropbox_file)

    if filename.endswith(".parquet"):
        data = pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        data = pd.read_csv(filename)
    else:
        raise ValueError("File must be either .parquet or .csv")

    os.remove(filename)

    return data


def dbx_load_geodataframe(filename, dropbox_path, crs, dropbox_app):

    dropbox_file = f"{dropbox_path}/{filename}"

    dropbox_app.files_download_to_file(filename, dropbox_file)

    data = geopandas.read_file(filename, crs=crs)
    os.remove(filename)

    return data


def dbx_load_metadata(dropbox_app):
    return dbx_load_dataframe(
        "join-roofs_images_obstacles.parquet",
        dropbox_path=DROPBOX_METADATA_PATH,
        dropbox_app=dropbox_app,
    )


def dbx_create_geo_metadata(dropbox_app):
    metadata = (
        dbx_load_metadata(dropbox_app=dropbox_app)
        .dropna(subset=["center_lng", "center_lat"])
        .rename(columns={"center_lng": "lon", "center_lat": "lat"})
    )

    return geopandas.GeoDataFrame(
        metadata,
        geometry=geopandas.points_from_xy(metadata.lon, metadata.lat),
        crs=4326,
    )


def dbx_load_geo_metadata(dropbox_app):
    return dbx_load_dataframe(
        "geometries-roofs_images_obstacles.parquet",
        dropbox_path=DROPBOX_METADATA_PATH,
        dropbox_app=dropbox_app,
    )


def dbx_load_earth(dropbox_app):
    return dbx_load_geodataframe(
        "earth.geo.json",
        dropbox_path=DROPBOX_EXTERNAL_DATA_PATH,
        crs=4326,
        dropbox_app=dropbox_app,
    )


def dbx_load_label_annotations(dropbox_app, update_annotations=False):

    if not update_annotations:
        return dbx_load_dataframe(
            "obstacles-labels_annotations.csv",
            dropbox_path=DROPBOX_ANNOTATIONS_PATH,
            dropbox_app=dropbox_app,
        )

    metadata_folder_contents = dbx.dropbox_list_contents_of(
        dropbox_app, DROPBOX_ANNOTATIONS_PATH
    )

    label_annotation_checkpoints = metadata_folder_contents.loc[
        lambda df: df.item_name.str.contains("-obstacles-labels_annotations.csv")
    ]

    dataframes = [
        dbx_load_dataframe(file, DROPBOX_ANNOTATIONS_PATH, dropbox_app)
        for file in label_annotation_checkpoints.item_name
    ]

    label_annotations = pd.concat(dataframes, ignore_index=True)

    annotated_data = (
        label_annotations.drop_duplicates(subset="roof_id", keep="last")
        .set_index("roof_id")
        .reset_index()
    )

    annotated_data.to_csv("obstacles-labels_annotations.csv", index=False)

    dbx.dropbox_upload_file_to(
        dropbox_app,
        "obstacles-labels_annotations.csv",
        f"{DROPBOX_ANNOTATIONS_PATH}/obstacles-labels_annotations.csv",
    )

    os.remove("obstacles-labels_annotations.csv")

    return annotated_data


def dbx_load_photo(
    photo_name, dropbox_folder, dropbox_app, greyscale_only: bool = False
):

    dropbox_path = f"{dropbox_folder}/{photo_name}"

    dropbox_app.files_download_to_file(photo_name, dropbox_path)

    if greyscale_only:
        greyscale_image = cv.imread(photo_name, 0)

        os.remove(photo_name)

        return greyscale_image

    bgr_image = cv.imread(photo_name, 1)
    greyscale_image = cv.imread(photo_name, 0)

    os.remove(photo_name)

    return bgr_image, greyscale_image


def dbx_load_photos_from_roof_id(
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


def load_and_crop_roof_from_roof_id(
    roof_id,
    photos_metadata,
    dropbox_path,
    dropbox_app,
    greyscale_only: bool = False,
):
    roof_px_coord, obstacles_px_coord = get_coordinates_from_roof_id(
        roof_id, photos_metadata
    )

    if greyscale_only:
        greyscale_image = dbx_load_photos_from_roof_id(
            roof_id, photos_metadata, dropbox_path, dropbox_app, greyscale_only
        )
        return rotate_and_crop_roof(greyscale_image, roof_px_coord)

    bgr_image, greyscale_image = dbx_load_photos_from_roof_id(
        roof_id, photos_metadata, dropbox_path, dropbox_app
    )

    k2_labelled_image = draw_boundaries(bgr_image, roof_px_coord, obstacles_px_coord)
    bgr_roof = rotate_and_crop_roof(k2_labelled_image, roof_px_coord)
    greyscale_roof = rotate_and_crop_roof(greyscale_image, roof_px_coord)

    return k2_labelled_image, bgr_roof, greyscale_roof
