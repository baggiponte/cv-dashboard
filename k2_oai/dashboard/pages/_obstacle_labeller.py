"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

from datetime import datetime

import numpy as np
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io import dropbox as dbx


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


def _load_previous_photo():
    st.session_state["roof_id_selector"] -= 1


def _load_next_photo():
    st.session_state["roof_id_selector"] += 1


def _save_labels_to_dropbox(
    labels_data=None,
    filename="roof-obstacles_labels_quality",
    upload_to="/k2/metadata/transformed_data",
):

    if labels_data is None:
        labels_data = st.session_state["label_quality_data"]

    dbx_app = utils.dbx_get_connection()

    timestamp = datetime.now().replace(microsecond=0)

    target_file_path = f"/tmp/{timestamp}-{filename}.csv"
    upload_path = f"{upload_to}/{timestamp}-{filename}.csv"

    (
        labels_data.loc[lambda df: df.label_quality.notna()].to_csv(
            target_file_path, index=False
        )
    )

    dbx.upload_file_to_dropbox(dbx_app, target_file_path, upload_path)


def obstacle_labeller_page():
    st.title(":mag: Labelling Tool")
    st.write(
        "Choose a roof id to see the labels that have been assigned to it",
        "then use the buttons provided below to mark the label as good (`Y`),",
        "bad (`N`) or maybe (`M`).",
    )

    # +------------------------------+
    # | Update Sidebar and Load Data |
    # +------------------------------+

    with st.sidebar:
        st.subheader("Data Source")

        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=[
                "small_photos-api_upload",
            ]
            + [f"large_photos-{i}K_{i+5}K-api_upload" for i in range(0, 10, 5)],
        )

        photos_metadata, dbx_photo_list = utils.dbx_get_photos_and_metadata(
            chosen_folder
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
        st.write("You can choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button("🔀", on_click=_load_random_photo, args=(roofs_to_label,))

        # roof ID selector
        # ----------------
        st.write("...or choose it manually:")
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

    buf, st_keep, st_drop, st_maybe, buf = st.columns((2, 1, 1, 1, 2))

    st_keep.button(
        "Keep label",
        help="Mark the label as good",
        on_click=_mark_photo,
        args=("Y",),
    )

    st_drop.button(
        "Drop label",
        help="Mark the label as bad",
        on_click=_mark_photo,
        args=("N",),
    )

    st_maybe.button(
        "Maybe",
        help="Do not mark the label and move on",
        on_click=_mark_photo,
        args=("M",),
    )

    # +---------------+
    # | Plot the roof |
    # +---------------+

    # Roof Only
    # ---------
    st.image(
        bgr_roof,
        use_column_width=True,
        channels="BGRA",
        caption="Roof with database labels",
    )

    # Photo Switcher
    # --------------

    # TODO: previous and next do not work:
    #       you cannot simply add or subtract 1 from the roof_id

    buf, st_previous, st_random, st_next, buf = st.columns((3, 1, 1, 1, 3))

    # st_previous.button(
    #     "⏪",
    #     help="Go to the previous roof",
    #     on_click=_load_previous_photo,
    # )

    st_random.button(
        "🔀",
        help="Go to a random roof",
        on_click=_load_random_photo,
        args=(roofs_to_label,),
    )

    # st_next.button(
    #     "⏩",
    #     help="Go to the next roof",
    #     on_click=_load_next_photo,
    # )

    # Full Photo
    # ----------
    with st.expander("View the original photo in full"):
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
        with st.expander("View the label quality dataset:"):
            st.dataframe(label_quality_data)

    if st_save.button("Save labels", help="Save the labels vetted so far to Dropbox"):
        _save_labels_to_dropbox()
