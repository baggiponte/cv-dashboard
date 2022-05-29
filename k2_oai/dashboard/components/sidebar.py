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
        for file in utils.st_list_contents_of(DROPBOX_RAW_PHOTOS_ROOT).item_name.values
        if not file.endswith(".csv")
    )

    chosen_folder = st.selectbox(
        "Select the folder to load the photos from:",
        options=photos_folders,
        index=0,
        key="photos_folder",
    )

    photos_metadata, photo_list = utils.st_load_photo_list_and_metadata(
        photos_folder=chosen_folder,
        photos_root_path=DROPBOX_RAW_PHOTOS_ROOT,
        geo_metadata=geo_metadata,
    )

    st.info(
        f"""
        Available photos: {photo_list.shape[0]}

        Unique roof ids: {photos_metadata.roof_id.unique().shape[0]}
        """
    )

    return chosen_folder, photos_metadata, photo_list


def config_annotations(mode: str):

    is_valid_method(mode, ["labels", "hyperparameters"])

    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    else:
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH

    st.markdown("## :card_index_dividers: Annotations")

    folder_contents = sorted(
        utils.st_list_contents_of(dropbox_path).item_name.to_list(),
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
