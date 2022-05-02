import os

import streamlit as st

from k2_oai.dashboard import pages
from k2_oai.io import dropbox as dbx


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
            st_oauth_text_boxes, oauth_result = dbx.st_dropbox_oauth2_connect()
            if oauth_result is not None:
                st.session_state["access_token"] = oauth_result.access_token
                st.session_state["refresh_token"] = oauth_result.refresh_token

    if st.session_state["access_token"] is not None:
        st_oauth_text_boxes.empty()

        readme_text = st.title(":house: Welcome!")

        st.sidebar.title(":gear: Settings")
        app_mode = st.sidebar.selectbox(
            "Which interface would you like to use?",
            ("Show Instructions", "Obstacle Detection", "Obstacle Labelling Tool"),
        )

        if app_mode == "Show Instructions":
            st.sidebar.success("Choose a mode from the sidebar to get started.")
        if app_mode == "Obstacle Labelling Tool":
            readme_text.empty()
            st.sidebar.markdown("___")
            pages.obstacle_labeller_page()
        elif app_mode == "Obstacle Detection":
            readme_text.empty()
            st.sidebar.markdown("___")
            pages.obstacle_detection_page()


if __name__ == "__main__":
    main()
