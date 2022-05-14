"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

import numpy as np
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io.dropbox_paths import DROPBOX_METADATA_PATH, DROPBOX_RAW_PHOTOS_ROOT

__all__ = ["obstacle_labeller_page"]


def annotate_labels(mark: str, roof_id, label_data):
    label_data.loc[label_data["roof_id"] == roof_id, "label_quality"] = mark


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
        photos_folders = utils.st_list_contents_of(DROPBOX_RAW_PHOTOS_ROOT).item_name

        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=photos_folders,
            index=0,
        )

        photos_metadata, dbx_photo_list = utils.st_load_photo_list_and_metadata(
            photos_folder=chosen_folder,
            photos_root_path=DROPBOX_RAW_PHOTOS_ROOT,
        )

        st.info(f"Available photos: {dbx_photo_list.shape[0]}")

        st.markdown("---")

    # +---------------------------------------------------+
    # | Cache DataFrame to store label_quality assessment |
    # +---------------------------------------------------+

    if "label_annotations" not in st.session_state:
        st.session_state["label_annotations"] = (
            photos_metadata[["roof_id", "imageURL"]]
            .drop_duplicates("roof_id")
            .sort_values("roof_id")
            .assign(label_annotations=np.NaN)
        )

    if "roofs_to_label" not in st.session_state:
        st.session_state["roofs_to_label"] = st.session_state["label_annotations"].loc[
            lambda df: df.label_quality.isna()
        ]

    label_annotations = st.session_state["label_annotations"]
    roofs_to_label = st.session_state["roofs_to_label"]

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:

        st.subheader("Label Annotations")

        # roof ID randomizer
        # ------------------

        st.write("Choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button("🔀", on_click=utils.load_random_photo, args=(roofs_to_label,))

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

    k2_labelled_image, bgr_roof, _ = utils.load_and_crop_roof_from_roof_id(
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
        help="Load a random roof",
        on_click=utils.load_random_photo,
        args=(roofs_to_label,),
    )
    st_keep.button(
        "Yes",
        help="Mark the label as good",
        on_click=annotate_labels,
        args=("Y", chosen_roof_id, label_annotations),
    )

    st_drop.button(
        "No",
        help="Mark the label as bad",
        on_click=annotate_labels,
        args=("N", chosen_roof_id, label_annotations),
    )

    st_maybe.button(
        "To improve",
        help="After some tweaking, label can be marked as good",
        on_click=annotate_labels,
        args=("M", chosen_roof_id, label_annotations),
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
        with st.expander("View the annotations:", expanded=True):
            st.dataframe(label_annotations.dropna(subset="label_annotations"))

    with st_save:
        st.info(
            f"Currently vetted {len(label_annotations.dropna(subset='label_annotations'))} roofs"  # noqa E50
        )
        if st.button("💾", help="Save the annotations done so far to Dropbox"):
            utils.save_annotations_to_dropbox(
                label_annotations.dropna(subset="label_annotations"),
                "obstacles-annotated_labels",
                DROPBOX_METADATA_PATH,
            )
