"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

import altair as alt
import numpy as np
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io.dropbox_paths import DROPBOX_ANNOTATIONS_PATH, DROPBOX_RAW_PHOTOS_ROOT

__all__ = ["obstacle_annotator_page"]


def annotate_labels(mark: str, roof_id, label_data):
    label_data.loc[label_data["roof_id"] == roof_id, "label_annotation"] = mark


def obstacle_annotator_page():
    st.title(":mag: Obstacle Annotation Tool")
    st.write(
        "Choose a roof id to see the labels that have been assigned to it",
        "then use the buttons provided below to mark the label as good (`Y`),",
        "bad (`N`) or to be improved (`M`).",
    )

    # +------------------------------+
    # | Update Sidebar and Load Data |
    # +------------------------------+

    with st.sidebar:

        st.markdown("## Data Sources")

        st.markdown("### Photo Folder")

        # get options for `chosen_folder`
        photos_folders = utils.st_list_contents_of(DROPBOX_RAW_PHOTOS_ROOT).item_name

        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=photos_folders,
            index=0,
            key="photos_folder",
        )

        photos_metadata, dbx_photo_list = utils.st_load_photo_list_and_metadata(
            photos_folder=chosen_folder,
            photos_root_path=DROPBOX_RAW_PHOTOS_ROOT,
        )

        st.info(f"Available photos: {dbx_photo_list.shape[0]}")

        st.markdown("### Annotations Data")

        annotations_folder = utils.st_list_contents_of(
            DROPBOX_ANNOTATIONS_PATH
        ).item_name.to_list()

        annotations_filename = st.selectbox(
            "Select the file to get and save the annotations: ",
            options=["New File"] + annotations_folder,
            index=1,
            key="annotations_filename",
        )

        st.markdown("---")

    if st.session_state["annotations_filename"] == "New File":
        st.session_state["label_annotations"] = (
            photos_metadata[["roof_id", "imageURL"]]
            .drop_duplicates("roof_id")
            .sort_values("roof_id")
            .assign(label_annotation=np.NaN)
        )
    else:
        st.session_state["label_annotations"] = utils.st_load_label_annotations(
            annotations_filename
        )

    if "roofs_still_to_annotate" not in st.session_state:
        st.session_state["roofs_still_to_annotate"] = photos_metadata.roof_id.loc[
            lambda df: ~df.isin(st.session_state["label_annotations"].roof_id)
        ]

    label_annotations = st.session_state["label_annotations"]
    roofs_still_to_annotate = st.session_state["roofs_still_to_annotate"]

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:

        st.subheader("Label Annotations")

        # roof ID randomizer
        # ------------------

        st.write("Choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button(
            "🔀",
            help="Get a random roof ID that was not labelled yet",
            on_click=utils.load_random_photo,
            args=(roofs_still_to_annotate,),
        )

        # roof ID selector
        # ----------------
        st.write("...or manually:")
        chosen_roof_id = st.selectbox(
            "Roof identifier:",
            options=photos_metadata.roof_id.unique(),
            help="Choose one out of all the available roof ids",
            key="roof_id_selector",
        )

        # uncomment if adding another section
        # st.markdown("---")

    k2_labelled_image, bgr_roof, _ = utils.load_and_crop_roof_from_roof_id(
        int(chosen_roof_id), photos_metadata, chosen_folder
    )

    # +-------------------+
    # | Labelling Actions |
    # +-------------------+

    st_count, st_randomizer, buf, st_keep, st_drop, st_maybe, buf = st.columns(
        (1.5, 1, 0.5, 1, 1, 1, 0.5)
    )

    with st_count:
        if chosen_roof_id not in label_annotations.roof_id:
            st.info(f"`{chosen_roof_id}` not annotated yet")
        else:
            st.warning(f"`{chosen_roof_id}` already annotated")

    st_randomizer.button(
        "🔀",
        help="Load a random roof ID that was not labelled yet",
        on_click=utils.load_random_photo,
        args=(roofs_still_to_annotate,),
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

    st_data, st_save = st.columns((5, 1))

    with st_data:
        with st.expander("View the annotations:", expanded=True):
            st.dataframe(label_annotations.dropna(subset="label_annotation"))

    with st_save:
        st.info(
            f"{len(label_annotations.dropna(subset='label_annotation'))} roofs vetted"
        )

        if annotations_filename == "obstacles-labels_annotations.csv":
            filename = "obstacles-labels_annotations"
        else:
            filename = "checkpoint-labels_annotations"

        if st.button("💾", help=f"Save annotations to {filename}.csv"):
            if annotations_filename == "New File":
                make_checkpoint = True
            else:
                make_checkpoint = False

            utils.save_annotations_to_dropbox(
                label_annotations.dropna(subset="label_annotation"),
                filename,
                DROPBOX_ANNOTATIONS_PATH,
                make_checkpoint,
            )

    # +------------------------+
    # | Annotations Statistics |
    # +------------------------+

    annotations = (
        label_annotations.groupby("label_annotation")
        .size()
        .rename(index={"Y": "Good", "N": "Bad", "M": "Improve"})
        .reset_index()
        .rename(columns={0: "Count", "label_annotation": "Annotation"})
        .sort_values("Count", ascending=False)
    )

    fig = (
        alt.Chart(annotations)
        .mark_bar()
        .encode(x="Annotation:N", y="Count:Q", tooltip=["Count"])
    )

    with st.sidebar:
        st.subheader("Annotations Statistics")
        st.altair_chart(fig, use_container_width=True)
