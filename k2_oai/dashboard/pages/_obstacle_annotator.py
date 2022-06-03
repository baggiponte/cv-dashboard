"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

from __future__ import annotations

import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.dashboard.components import buttons, sidebar

__all__ = ["obstacle_annotator_page"]


def obstacle_annotator_page(session_state_key="label_annoatations", mode="labels"):
    st.title(":mag: Obstacle Annotation Tool")

    with st.sidebar:

        # +---------------------+
        # | select data sources |
        # +---------------------+

        chosen_folder, obstacles_metadata, photo_list = sidebar.config_photo_folder()

        sidebar.count_duplicates(obstacles_metadata, photo_list)

        chosen_annotations_file = sidebar.config_annotations(mode=mode)

        annotated_roofs, remaining_roofs, all_annotations = sidebar.config_cache(
            session_state_key=session_state_key,
            metadata=obstacles_metadata,
            annotations_file=chosen_annotations_file,
        )

        chosen_roof_id = buttons.choose_roof_id(obstacles_metadata, remaining_roofs)

        # +-------------------+
        # | labelling actions |
        # +-------------------+

        st.markdown("## :pencil: Mark the annotations")

        st.info(f"Roofs annotated so far: {annotated_roofs.shape[0]}")

        is_trainable = st.radio(
            label="Can the photo be used for training?",
            options=[0, 1],
            index=0,
            help="0 means no, 1 means yes",
        )

        is_roof = st.radio(
            label="Does the label depict a roof?",
            options=[0, 1],
            index=0,
            help="e.g. grass was labelled instead of a roof",
        )

        roof_well_cropped = st.radio(
            label="Is the roof well cropped?",
            options=[0, 1],
            index=0,
            help="e.g. the cropped portion is smaller than the roof",
        )

        obstacles_well_cropped = st.radio(
            label="Are the obstacles well cropped?",
            options=[0, 1],
            index=0,
            help="e.g. the label is larger than the real obstacle",
        )

        all_obstacles_found = st.radio(
            label="Are all obstacles labelled?",
            options=[0, 1],
            index=0,
        )

        not_an_obstacle = st.radio(
            label="Was something other than an obstacle labelled?",
            options=[0, 1],
            index=0,
        )

        annotations = {
            "is_trainable": is_trainable,
            "is_roof": is_roof,
            "roof_well_cropped": roof_well_cropped,
            "obstacles_well_cropped": obstacles_well_cropped,
            "all_obstacles_found": all_obstacles_found,
            "not_an_obstacle": not_an_obstacle,
        }

        sidebar.write_and_save_annotations(
            new_annotations=annotations,
            annotations_data=all_annotations,
            annotations_savefile=chosen_annotations_file,
            roof_id=chosen_roof_id,
            folder=chosen_folder,
            metadata=obstacles_metadata,
            session_state_key=session_state_key,
            mode=mode,
        )

    # +------------------------+
    # | Load and plot the roof |
    # +------------------------+

    if chosen_roof_id in all_annotations.roof_id.values:
        st.info(f"Roof {chosen_roof_id} is already annotated")
    else:
        st.warning(f"Roof {chosen_roof_id} is not annotated")

    photo, roof, labelled_photo, labelled_roof = utils.st_load_photo_and_roof(
        int(chosen_roof_id),
        obstacles_metadata,
        chosen_folder,
    )

    st_labelled, st_not_labelled = st.columns(2)

    with st_labelled:
        st.image(
            labelled_roof,
            use_column_width=True,
            channels="BGR",
            caption="Labelled Roof",
        )

        st.image(
            labelled_photo,
            use_column_width=True,
            channels="BGR",
            caption="Cropped roof",
        )

    with st_not_labelled:
        st.image(
            roof,
            use_column_width=True,
            channels="BGR",
            caption="Original image, labelled",
        )

        st.image(
            photo,
            use_column_width=True,
            channels="BGR",
            caption="Original image",
        )

    # +----------------+
    # | Roof metadata  |
    # +----------------+

    with st.expander(f"Roof {chosen_roof_id} metadata:"):
        st.dataframe(
            obstacles_metadata.loc[obstacles_metadata.roof_id == chosen_roof_id]
        )

    with st.expander("View the annotations:", expanded=True):
        st.dataframe(all_annotations)
