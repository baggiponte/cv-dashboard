import os

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
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    st.session_state['access_token'] = os.environ.get("DROPBOX_ACCESS_TOKEN")
else:
    if st.session_state['access_token'] is None:
        placeholders_list, oauth_result = dropbox_connect_oauth2_streamlit()
        if oauth_result is not None:
            st.session_state['access_token'] = oauth_result.access_token
            st.session_state['refresh_token'] = oauth_result.refresh_token

if st.session_state['access_token'] is not None:

    for placeholder in placeholders_list:
        placeholder.empty()

    obstacle_detection_page()

elif st.session_state['access_token'] is None:

    pass

else:

    st.error("Invalid access token!")

