import os

import streamlit as st

from k2_oai.dashboard.pages import obstacle_detection_page
from k2_oai.io import dropbox as dbx

st.set_page_config(page_title="K2 <-> OAI", layout="wide", initial_sidebar_state="auto")

st.title("Obstacle Detection Dashboard")

if "access_token" not in st.session_state:
    st.session_state["access_token"] = None

st_oauth_text_boxes = list()
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    st.session_state["access_token"] = os.environ.get("DROPBOX_ACCESS_TOKEN")
else:
    if st.session_state["access_token"] is None:
        st_oauth_text_boxes, oauth_result = dbx.st_dropbox_oauth2_connect()
        if oauth_result is not None:
            st.session_state["access_token"] = oauth_result.access_token
            st.session_state["refresh_token"] = oauth_result.refresh_token

if st.session_state["access_token"] is not None:
    for st_text_box in st_oauth_text_boxes:
        st_text_box.empty()

    obstacle_detection_page()

elif st.session_state["access_token"] is None:
    pass

else:
    st.error("Invalid access token!")
