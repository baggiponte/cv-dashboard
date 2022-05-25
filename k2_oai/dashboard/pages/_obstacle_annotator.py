"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from k2_oai.dashboard import components, utils
from k2_oai.io.dropbox_paths import DROPBOX_LABEL_ANNOTATIONS_PATH

__all__ = ["obstacle_annotator_page"]


def annotate_labels(marks, roof_id, photos_folder, photos_metadata):
    image_url = photos_metadata.loc[
        photos_metadata["roof_id"] == roof_id, "imageURL"
    ].values[0]

    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    new_row = pd.DataFrame(
        [
            {
                "roof_id": roof_id,
                "annotation": marks,
                "imageURL": image_url,
                "photos_folder": photos_folder,
                "annotation_time": now,
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


def load_random_photo(roofs_list):
    st.session_state["roof_id"] = np.random.choice(roofs_list)


def change_roof_id(how: str, available_roofs):
    # note: np.where returns a tuple, in this case ([array],).
    # use the double indexing like [0][0]!
    current_index: int = np.where(available_roofs == st.session_state["roof_id"])[0][0]

    if how == "next":
        # otherwise is out of index
        if current_index < len(available_roofs) - 1:
            st.session_state["roof_id"] = available_roofs[current_index + 1]
    elif how == "previous":
        if current_index > 0:
            st.session_state["roof_id"] = available_roofs[current_index - 1]
    else:
        raise ValueError(f"Invalid `how`: {how}. Must be `next` or `previous`.")


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
        ) = components.sidebar_chose_photo_folder()

        # +--------------------------------------+
        # | Choose where to save the annotations |
        # +--------------------------------------+

        st.markdown("## :pencil: Annotations Data")

        annotations_folder = sorted(
            utils.st_list_contents_of(
                DROPBOX_LABEL_ANNOTATIONS_PATH
            ).item_name.to_list(),
            reverse=True,
        )

        chosen_annotations_file = st.selectbox(
            "Select the file to get and save the annotations: ",
            options=["New File"] + annotations_folder,
            index=0,
            key="annotations_filename",
        )

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

    if chosen_annotations_file == "New File":
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
            Annotated: {annotated_roofs.shape[0]}.
            To annotate: {roofs_left_to_annotate.shape[0]}
            """
        )
        st.markdown("---")

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:

        st.subheader("Label Annotations")

        # roof ID randomizer
        # ------------------

        st.write("Choose a roof ID randomly...")

        buf, st_previous, st_rand, st_next, buf = st.columns((0.5, 1, 1, 1, 0.5))

        st_previous.button(
            "⬅️",
            help="Load the photo before this one. "
            "If nothing happens, this is the first photo.",
            on_click=change_roof_id,
            args=("previous", available_roofs),
            key="sidebar_previous_roof_id",
        )

        st_rand.button(
            "🔀",
            help="Load a random photo that was not labelled yet",
            on_click=load_random_photo,
            args=(roofs_left_to_annotate,),
            key="sidebar_random_roof_id",
        )

        st_next.button(
            "➡️",
            help="Load the photo right after this one. "
            "If nothing happens, this is the last photo.",
            on_click=change_roof_id,
            args=("next", available_roofs),
            key="sidebar_next_roof_id",
        )

        # roof ID selector
        # ----------------
        st.write("...or manually:")
        chosen_roof_id = st.selectbox(
            "Roof identifier:",
            options=available_roofs,
            help="Choose one out of all the available roof ids",
            key="roof_id",
        )

    k2_labelled_image, bgr_roof, _ = utils.load_and_crop_roof_from_roof_id(
        int(chosen_roof_id), photos_metadata, chosen_folder
    )

    # +-------------------+
    # | Labelling Actions |
    # +-------------------+

    (
        buf,
        st_previous,
        st_randomizer,
        st_next,
        buf,
        st_mark,
        st_annotations,
        st_save,
    ) = st.columns((0.5, 1, 1, 1, 0.3, 1, 4, 1))

    st_previous.button(
        "⬅️",
        help="Load the previous photo. If nothing happens, this is the first photo.",
        on_click=change_roof_id,
        args=("previous", available_roofs),
        key="previous_roof_id",
    )

    st_randomizer.button(
        "🔀",
        help="Load a random photo that was not labelled yet",
        on_click=load_random_photo,
        args=(roofs_left_to_annotate,),
        key="random_roof_id",
    )

    st_next.button(
        "➡️",
        help="Load the next photo. If nothing happens, this is the last photo.",
        on_click=change_roof_id,
        args=("next", available_roofs),
        key="next_roof_id",
    )

    chosen_annotations = st_annotations.radio(
        label="Can the photo be used for training?",
        options=[0, 1],
        index=0,
        help="0 means no, 1 means yes",
        key="binary_annotation",
    )

    st_mark.button(
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

    with st_save:
        filename = (
            "obstacles-labels_annotations"
            if chosen_annotations_file == "obstacles-labels_annotations.csv"
            else "checkpoint-labels_annotations"
        )

        make_checkpoint = True if chosen_annotations_file == "New File" else False

        if st.button("💾", help=f"Save annotations to {filename}.csv"):
            utils.save_annotations_to_dropbox(
                all_annotations,
                filename,
                DROPBOX_LABEL_ANNOTATIONS_PATH,
                make_checkpoint,
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
