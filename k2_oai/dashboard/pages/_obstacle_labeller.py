"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

from datetime import datetime

import numpy as np
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io import dropbox as dbx

_DBX_PHOTOS_PATH = "/k2/raw_photos"
_DBX_LABELS_PATH = "/k2/metadata/transformed_data"


def _load_random_photo(roofs_list=None):
    if roofs_list is None:
        roofs_list = st.session_state["roofs_to_label"]
    st.session_state["roof_id_selector"] = np.random.choice(roofs_list.roof_id)


def _mark_photo(mark: str, roof_id=None, label_data=None):
    if roof_id is None:
        roof_id = st.session_state["roof_id_selector"]

    if label_data is None:
        label_data = st.session_state["label_quality_data"]

    label_data.loc[label_data["roof_id"] == roof_id, "label_quality"] = mark


def _save_labels_to_dropbox(
    labels_data=None,
    export_filename="obstacles-labels_quality",
    dbx_root_path=None,
    to_temp: bool = True,
):

    dbx_root_path = dbx_root_path or _DBX_LABELS_PATH

    dbx_app = utils.dbx_get_connection()

    if labels_data is None:
        labels_data = st.session_state["label_quality_data"]

    data_to_export = labels_data.loc[lambda df: df.label_quality.notna()]

    if to_temp:
        timestamp = datetime.now().replace(microsecond=0).strftime("%Y_%m_%d-%H_%M_%S")
        filename = f"{timestamp}-{export_filename}.csv"
    else:
        filename = f"{export_filename}.csv"

    upload_path = f"{dbx_root_path}/{filename}"

    data_to_export.to_csv(f"/tmp/{filename}", index=False)

    dbx.upload_file_to_dropbox(
        dbx_app, file_path_from=f"/tmp/{filename}", file_path_to=upload_path
    )


def obstacle_labeller_page():
    st.title(":mag: Labelling Tool")
    st.write(
        "Choose a roof id to see the labels that have been assigned to it",
        "then use the buttons provided below to mark the label as good (`Y`),",
        "bad (`N`) or to be improved (`M`).",
    )

    # +------------------------------+
    # | Update Sidebar and Load Data |
    # +------------------------------+

    with st.sidebar:

        st.subheader("Data Source")

        # get options for `chosen_folder`
        photos_folders = utils.dbx_list_dir_contents(_DBX_PHOTOS_PATH).item_name

        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=photos_folders,
            index=3,
        )

        photos_metadata, dbx_photo_list = utils.dbx_get_photo_list_and_metadata(
            photos_folder=chosen_folder,
            photos_root_path=_DBX_PHOTOS_PATH,
        )

        st.info(f"Available photos: {dbx_photo_list.shape[0]}")

        st.markdown("---")

    # +---------------------------------------------------+
    # | Cache DataFrame to store label_quality assessment |
    # +---------------------------------------------------+

    if "label_quality_data" not in st.session_state:
        st.session_state["label_quality_data"] = (
            photos_metadata[["roof_id", "imageURL"]]
            .drop_duplicates("roof_id")
            .sort_values("roof_id")
            .assign(label_quality=np.NaN)
        )

    if "roofs_to_label" not in st.session_state:
        st.session_state["roofs_to_label"] = st.session_state["label_quality_data"].loc[
            lambda df: df.label_quality.isna()
        ]

    label_quality_data = st.session_state["label_quality_data"]
    roofs_to_label = st.session_state["roofs_to_label"]

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:

        st.subheader("Label Quality Assessment")

        # roof ID randomizer
        # ------------------

        st.write("Choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button("🔀", on_click=_load_random_photo, args=(roofs_to_label,))

        # roof ID selector
        # ----------------
        st.write("...or manually:")
        chosen_roof_id = st.selectbox(
            "Roof identifier:",
            options=roofs_to_label,
            help="Identifier of the roof, whose label we want to inspect.",
            key="roof_id_selector",
        )

        st.markdown("---")

    k2_labelled_image, bgr_roof, _ = utils.crop_roofs_from_roof_id(
        int(chosen_roof_id), photos_metadata, chosen_folder
    )

    # +-------------------+
    # | Labelling Actions |
    # +-------------------+

    st_count, st_randomizer, buf, st_keep, st_drop, st_maybe, buf = st.columns(
        (1, 1, 0.5, 1, 1, 1, 0.5)
    )

    st_count.markdown(f"Roof ID: `{chosen_roof_id}`")

    st_randomizer.button(
        "🔀",
        help="Go to a random roof",
        on_click=_load_random_photo,
        args=(roofs_to_label,),
    )
    st_keep.button(
        "Yes",
        help="Mark the label as good",
        on_click=_mark_photo,
        args=("Y",),
    )

    st_drop.button(
        "No",
        help="Mark the label as bad",
        on_click=_mark_photo,
        args=("N",),
    )

    st_maybe.button(
        "To improve",
        help="The label will become good after improvements",
        on_click=_mark_photo,
        args=("M",),
    )

    # +---------------+
    # | Plot the roof |
    # +---------------+

    st_roof_photo, st_full_photo = st.columns(2)

    with st_roof_photo:
        st.image(
            bgr_roof,
            use_column_width=True,
            channels="BGRA",
            caption="Roof with database labels",
        )

    with st_full_photo:
        st.image(
            k2_labelled_image,
            use_column_width=True,
            channels="BGRA",
            caption="Original image",
        )

    # View Label Quality Dataset
    # --------------------------

    st_data, st_save = st.columns((6, 1))

    with st_data:
        with st.expander("View the label quality dataset:", expanded=True):
            st.dataframe(label_quality_data.dropna(subset="label_quality"))

    with st_save:
        st.info(
            f"Currently vetted {len(label_quality_data.dropna(subset='label_quality'))} roofs"  # noqa E50
        )
        if st.button("💾", help="Save the labels vetted so far to Dropbox"):
            _save_labels_to_dropbox()
