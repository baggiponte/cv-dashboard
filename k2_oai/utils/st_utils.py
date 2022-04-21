from k2_oai.utils._draw_boundaries import *

from k2_oai.utils.dropbox_io_utils import *


@st.cache(allow_output_mutation=True)
def get_dbx_connection():
    return dropbox_connect(st.session_state['access_token'], st.session_state['refresh_token'])


@st.cache
def get_metadata_df():
    _dbx = get_dbx_connection()
    _dbx.files_download_to_file(
        "inner_join-roofs_images_obstacles.csv",
        "/k2/metadata/raw_data/inner_join-roofs_images_obstacles.csv"
    )
    metadata_df = pd.read_csv("inner_join-roofs_images_obstacles.csv")
    return metadata_df


@st.cache
def get_photos_list(folder_name):
    _dbx = get_dbx_connection()
    st_dropbox_list_files_df = get_dropbox_list_files_df(_dbx, "/k2/raw_photos/{}".format(folder_name))
    return st_dropbox_list_files_df


@st.cache
def get_remote_data_structures(folder_name):
    _dbx = get_dbx_connection()
    st_dropbox_list_files_df = get_dropbox_list_files_df(_dbx, "/k2/raw_photos/{}".format(folder_name))
    _dbx.files_download_to_file(
        "inner_join-roofs_images_obstacles.csv",
        "/k2/metadata/raw_data/inner_join-roofs_images_obstacles.csv"
    )
    metadata_df = pd.read_csv("inner_join-roofs_images_obstacles.csv")
    return st_dropbox_list_files_df, metadata_df


@st.cache(allow_output_mutation=True)
def load_photo_from_remote(folder_name, photo_name):
    _dbx = get_dbx_connection()
    _dbx.files_download_to_file(
        photo_name, "/k2/raw_photos/{}/{}".format(folder_name, photo_name)
    )
    im_bgr: np.ndarray = cv.imread(photo_name, 1)
    im_gs: np.ndarray = cv.imread(photo_name, 0)
    return im_bgr, im_gs
