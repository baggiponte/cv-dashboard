"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.dashboard.components import buttons, sidebar
from k2_oai.io.dropbox_paths import DROPBOX_LABEL_ANNOTATIONS_PATH

__all__ = ["obstacle_annotator_page"]


def annotate_labels(marks, roof_id, photos_folder, photos_metadata):
    image_url = photos_metadata.loc[
        photos_metadata["roof_id"] == roof_id, "imageURL"
    ].values[0]

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    new_row = pd.DataFrame(
        [
            {
                "roof_id": roof_id,
                "annotation": marks,
                "imageURL": image_url,
                "photos_folder": photos_folder,
                "annotation_time": timestamp,
            }
        ]
    ).astype({"roof_id": int, "annotation": int})

    st.session_state["label_annotations"] = (
        pd.concat([st.session_state["label_annotations"], new_row], ignore_index=True)
        .astype({"roof_id": int, "annotation": int})
        .drop_duplicates(subset=["roof_id"], keep="last")
        .sort_values("roof_id")
        .reset_index(drop=True)
    )


def obstacle_annotator_page():
    st.title(":mag: Obstacle Annotation Tool")
    st.write()

    with st.sidebar:

        # +---------------------------------------------+
        # | Load photos metadata from a selected folder |
        # +---------------------------------------------+

        (
            chosen_folder,
            photos_metadata,
            photo_list,
        ) = sidebar.config_photo_folder()

        # +---------------------------------------+
        # | Load an existing annotations savefile |
        # +---------------------------------------+

        (
            chosen_annotations_file,
            chosen_savefile,
            use_checkpoints,
        ) = sidebar.config_annotations(mode="labels")

    # +---------------------------------+
    # | Label annotation stats and data |
    # +---------------------------------+

    available_roofs = photos_metadata.roof_id.unique()

    if "label_annotations" not in st.session_state:
        st.session_state["label_annotations"] = (
            pd.DataFrame()
            .assign(
                roof_id=np.NaN,
                imageURL=np.NaN,
                photos_folder=np.NaN,
                annotation=np.NaN,
                annotation_time=np.NaN,
            )
            .astype({"roof_id": int, "annotation": int})
        )

    annotated_roofs = (
        st.session_state["label_annotations"]
        .dropna(subset=["annotation"])
        .roof_id.values
    )

    roofs_left_to_annotate = photos_metadata.loc[
        lambda df: ~df.roof_id.isin(annotated_roofs),
        "roof_id",
    ].unique()

    if chosen_annotations_file == "New Checkpoint":
        all_annotations = st.session_state["label_annotations"].dropna(
            subset="annotation"
        )
    else:
        existing_annotations = utils.st_load_annotations(chosen_annotations_file)

        roofs_left_to_annotate = photos_metadata.loc[
            lambda df: ~df.roof_id.isin(existing_annotations.roof_id), "roof_id"
        ].unique()

        all_annotations = (
            pd.concat(
                [
                    existing_annotations,
                    st.session_state["label_annotations"].dropna(subset="annotation"),
                ],
                ignore_index=True,
            )
            .sort_values("roof_id")
            .reset_index(drop=True)
            .drop_duplicates(subset=["roof_id", "annotation"], keep="last")
        )

    with st.sidebar:
        st.info(
            f"""
            Roofs annotated so far: {annotated_roofs.shape[0]}
            """
        )
        st.markdown("---")

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:
        chosen_roof_id = buttons.choose_roof_id(available_roofs, roofs_left_to_annotate)

    k2_labelled_image, bgr_roof, _ = utils.load_and_crop_roof_from_roof_id(
        int(chosen_roof_id), photos_metadata, chosen_folder
    )

    # +-------------------+
    # | Labelling Actions |
    # +-------------------+

    with st.sidebar:

        st.markdown("## :lower_left_crayon: Mark the annotations")

        buf, st_annotate, buf, st_save, buf = st.columns((1.2, 1, 0.2, 1, 1.5))

        chosen_annotations = st.radio(
            label="Can the photo be used for training?",
            options=[0, 1],
            index=0,
            help="0 means no, 1 means yes",
            key="binary_annotation",
        )

        st_annotate.button(
            "📝",
            help="Write the annotations to the dataset",
            on_click=annotate_labels,
            args=(
                chosen_annotations,
                chosen_roof_id,
                chosen_folder,
                photos_metadata,
            ),
        )

        filename = utils.make_filename(chosen_savefile, use_checkpoints)
        if st_save.button("💾", help=f"Save annotations to {filename}"):
            utils.save_annotations_to_dropbox(
                all_annotations,
                filename,
                DROPBOX_LABEL_ANNOTATIONS_PATH,
            )

    if chosen_roof_id in all_annotations.roof_id.values:
        st.info(f"Roof {chosen_roof_id} is already annotated")
    else:
        st.warning(f"Roof {chosen_roof_id} is not annotated")

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

    # +----------------+
    # | Roof metadata  |
    # +----------------+

    with st.expander(f"Roof {chosen_roof_id} metadata:"):
        st.dataframe(photos_metadata.loc[photos_metadata.roof_id == chosen_roof_id])

    with st.expander("View the annotations:", expanded=True):
        st.dataframe(all_annotations)
