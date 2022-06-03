"""
Dashboard components to insert elements, preferably into the sidebar
"""
from typing import Any

import pandas as pd
import streamlit as st
from pandas import DataFrame

from k2_oai.dashboard import utils
from k2_oai.io.dropbox_paths import (
    DROPBOX_HYPERPARAM_ANNOTATIONS_PATH,
    DROPBOX_LABEL_ANNOTATIONS_PATH,
    DROPBOX_RAW_PHOTOS_ROOT,
)

__all__ = ["config_photo_folder", "config_annotations"]


def config_photo_folder(geo_metadata: bool = False, only_folders: bool = True):

    st.markdown("## :open_file_folder: Photos Folder")

    # get options for `chosen_folder`
    root_contents = utils.st_listdir_no_cache(DROPBOX_RAW_PHOTOS_ROOT).item_name.values
    photos_folders = sorted(file for file in root_contents if not file.endswith(".csv"))

    if not only_folders:
        photos_folders.insert(0, None)

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


def count_duplicates(obstacles_metadata, photo_list):

    total_obst = obstacles_metadata.pixelCoordinates_obstacle.notna().shape[0]
    unique_obst = obstacles_metadata.drop_duplicates(
        subset=["imageURL", "pixelCoordinates_obstacle"]
    ).shape[0]

    roofs_metadata = obstacles_metadata.drop_duplicates(subset="roof_id")
    total_roofs = roofs_metadata.pixelCoordinates_roof.notna().shape[0]
    unique_roofs = roofs_metadata.drop_duplicates(
        subset=["imageURL", "pixelCoordinates_roof"]
    ).shape[0]

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

    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    elif mode == "hyperparameters" or mode == "hyperparams":
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH
    else:
        raise ValueError(f"Invalid {mode = }. Must be either 'labels' or 'hyperparams'")

    st.markdown("## :card_index_dividers: Annotations")

    folder_contents = sorted(
        utils.st_listdir_no_cache(dropbox_path).item_name.to_list(),
        reverse=True,
    )

    options_files = [file for file in folder_contents if "OLD-" not in file]

    annotations_file = st.selectbox(
        "Load existing annotations or create a new one",
        options=["New Checkpoint"] + options_files,
        index=0,
    )

    return annotations_file


def config_cache(session_state_key, metadata, annotations_file):

    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = pd.DataFrame(
            columns=["roof_id", "annotation_time"]
        )

    session_state = st.session_state[session_state_key]

    annotated_roofs = session_state.dropna(subset=["annotation_time"]).roof_id.values

    remaining_roofs = metadata.roof_id.loc[
        lambda df: ~df.isin(annotated_roofs)
    ].unique()

    if annotations_file == "New Checkpoint":
        all_annotations = session_state.dropna(subset="annotation_time")
    else:
        existing_annotations = utils.st_load_annotations(annotations_file)

        remaining_roofs = metadata.roof_id.loc[
            lambda df: ~df.isin(existing_annotations.roof_id.unique())
        ].unique()

        all_annotations = (
            pd.concat(
                [
                    existing_annotations,
                    session_state.dropna(subset="annotation_time"),
                ],
                ignore_index=True,
            )
            .sort_values("roof_id")
            .reset_index(drop=True)
            .drop_duplicates(subset=["roof_id", "annotation_time"], keep="last")
        )

    return annotated_roofs, remaining_roofs, all_annotations


def write_and_save_annotations(
    new_annotations: dict[str, Any],
    annotations_data: DataFrame,
    annotations_savefile: str,
    roof_id: int,
    folder: str,
    metadata: DataFrame,
    session_state_key: str,
    mode: str,
):

    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    elif mode == "hyperparameters" or mode == "hyperparams":
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'labels' or 'hyperparameters'")

    buf, st_annotate, buf, st_save, buf = st.columns((1.2, 1, 0.2, 1, 1.5))

    st_annotate.button(
        "📝",
        help="Write the annotations to the dataset",
        on_click=utils.annotate_labels,
        args=(new_annotations, session_state_key, roof_id, folder, metadata, mode),
    )

    savefile = st.text_input(
        label="Name of the file to save new annotations to:",
        value=annotations_savefile,
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

    filename = utils.make_filename(savefile, use_checkpoints)

    if st_save.button("💾", help=f"Save annotations to {filename}"):
        utils.st_save_annotations(
            annotations_data,
            filename,
            dropbox_path,
        )
