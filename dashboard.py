import streamlit as st
import matplotlib.pyplot as plt

from k2_oai.image_segmentation import *
from k2_oai.utils._draw_boundaries import *
from k2_oai.metrics import surface_absolute_error

from k2_oai.utils.dropbox_io_utils import *

from k2_oai.dashboard_pages.obstacle_detection import *


st.set_page_config(
    page_title="K2 <-> OAI",
    layout='wide',
    initial_sidebar_state='auto'
)

st.title("Obstacle detection dashboard")

if 'access_token' not in st.session_state:
    st.session_state['access_token'] = None

placeholders_list = list()
if st.session_state['access_token'] is None:
    placeholders_list, oauth_result = dropbox_connect_oauth2_streamlit()
    if oauth_result is not None:
        st.session_state['access_token'] = oauth_result.access_token
        st.session_state['refresh_token'] = oauth_result.refresh_token

if st.session_state['access_token'] is not None:

    obstacle_detection_page(placeholders_list)

elif st.session_state['access_token'] is None:

    pass

else:

    st.error("Invalid access token!")

