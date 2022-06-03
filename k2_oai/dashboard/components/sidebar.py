"""
Dashboard components to insert elements, preferably into the sidebar
"""

import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io.dropbox_paths import (
    DROPBOX_HYPERPARAM_ANNOTATIONS_PATH,
    DROPBOX_LABEL_ANNOTATIONS_PATH,
    DROPBOX_RAW_PHOTOS_ROOT,
)
from k2_oai.utils import is_valid_method

__all__ = ["config_photo_folder", "config_annotations"]


def config_photo_folder(geo_metadata: bool = False):

    st.markdown("## :open_file_folder: Photos Folder")

    # get options for `chosen_folder`
    photos_folders = sorted(
        file
        for file in utils.st_listdir(DROPBOX_RAW_PHOTOS_ROOT).item_name.values
        if not file.endswith(".csv")
    )

    chosen_folder = st.selectbox(
        "Select the folder to load the photos from:",
        options=photos_folders,
        index=0,
        key="photos_folder",
    )

    obstacles_metadata, photo_list = utils.st_load_photo_list_and_metadata(
        photos_folder=chosen_folder,
        geo_metadata=geo_metadata,
    )

    return chosen_folder, obstacles_metadata, photo_list


def obstacles_counts(obstacles_metadata, photo_list):

    total_obst = obstacles_metadata.pixelCoordinates_obstacle.notna().shape[0]
    unique_obst = obstacles_metadata.pixelCoordinates_obstacle.unique().shape[0]

    roofs_metadata = obstacles_metadata.drop_duplicates(subset="roof_id")
    total_roofs = roofs_metadata.pixelCoordinates_roof.notna().shape[0]
    unique_roofs = roofs_metadata.pixelCoordinates_roof.unique().shape[0]

    st.info(
        f"""
        Photos with tagged obstacles: {photo_list.shape[0]}

        Total roofs: {total_roofs}

        Total obstacles: {total_obst}
        """
    )

    st.warning(
        f"""
        Duplicate roofs: {total_roofs - unique_roofs}
        ({(total_roofs - unique_roofs) / total_roofs * 100:.2f}%)

        Duplicate obstacles: {total_obst - unique_obst}
        ({(total_obst - unique_obst) / total_obst * 100:.2f}%)
        """
    )


def config_annotations(mode: str):

    is_valid_method(mode, ["labels", "hyperparameters"])

    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    else:
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH

    st.markdown("## :card_index_dividers: Annotations")

    folder_contents = sorted(
        utils.st_listdir(dropbox_path).item_name.to_list(),
        reverse=True,
    )

    options_files = [file for file in folder_contents if "OLD-" not in file]

    annotations_file = st.selectbox(
        "Load existing annotations or create a new one",
        options=["New Checkpoint"] + options_files,
        index=0,
        key="annotations_filename",
    )

    savefile = st.text_input(
        label="Name of the file to save new annotations to:",
        value=annotations_file,
        help="Defaults to the same filename as the file you load "
        "- i.e. it will be overwritten. Specifying '.csv' is not necessary",
        key="savefile_name",
    )

    use_checkpoints = st.radio(
        label="Save the annotations as a checkpoint",
        options=[True, False],
        index=1,
        help="The file will be saved with a timestamp",
        key="use_checkpoints",
    )

    return annotations_file, savefile, use_checkpoints
