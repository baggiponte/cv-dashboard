import os

import streamlit as st

from k2_oai.dashboard import pages, utils


def main():
    st.set_page_config(
        page_title="K2 <-> OAI",
        page_icon=":house:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None

    st_oauth_text_boxes = st.empty()
    if st.session_state["access_token"] is None:
        if "DROPBOX_ACCESS_TOKEN" in os.environ:
            st.session_state["access_token"] = os.environ.get("DROPBOX_ACCESS_TOKEN")
        else:
            st_oauth_text_boxes, oauth_result = utils.st_dropbox_oauth2_connect()
            if oauth_result is not None:
                st.session_state["access_token"] = oauth_result.access_token
                st.session_state["refresh_token"] = oauth_result.refresh_token

    if st.session_state["access_token"] is not None:
        st_oauth_text_boxes.empty()

        readme_text = st.markdown(
            """
        # :house: Welcome!

        This is OAI's dashboard to explore the image segmentation models designed to
        detect obstacles on satellite images. The dashboard has the following modes:

        * `Instructions` is this page.
        * `Data Explorer` is an interface to perform exploratory data analysis.
        * `Obstacle Labelling Tool` us a tool to annotate images and create new labels.
        * `Obstacle Detection` is the interface to explore the image segmentation model.

        """
        )

        st.sidebar.title(":gear: Settings")
        app_mode = st.sidebar.selectbox(
            "Which interface would you like to use?",
            (
                "Instructions",
                "Data Explorer",
                "Obstacle Labelling Tool",
                "Obstacle Detection",
            ),
        )

        if app_mode == "Instructions":
            st.sidebar.success("Choose a mode from the sidebar to get started.")
        elif app_mode == "Obstacle Detection":
            readme_text.empty()
            st.sidebar.markdown("___")
            pages.obstacle_detection_page()
        elif app_mode == "Obstacle Labelling Tool":
            readme_text.empty()
            st.sidebar.markdown("___")
            pages.obstacle_labeller_page()
        elif app_mode == "Data Explorer":
            readme_text.empty()
            st.sidebar.markdown("___")
            pages.metadata_explorer_page()


if __name__ == "__main__":
    main()
