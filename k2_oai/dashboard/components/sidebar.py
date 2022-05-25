import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io.dropbox_paths import DROPBOX_RAW_PHOTOS_ROOT


def sidebar_chose_photo_folder(geo_metadata: bool = False):

    st.markdown("## :open_file_folder: Photos Folder")

    # get options for `chosen_folder`
    photos_folders = sorted(
        file
        for file in utils.st_list_contents_of(DROPBOX_RAW_PHOTOS_ROOT).item_name.values
        if not file.endswith(".csv")
    )

    chosen_folder = st.selectbox(
        "Select the folder to load the photos from:",
        options=photos_folders,
        index=0,
        key="photos_folder",
    )

    photos_metadata, photo_list = utils.st_load_photo_list_and_metadata(
        photos_folder=chosen_folder,
        photos_root_path=DROPBOX_RAW_PHOTOS_ROOT,
        geo_metadata=geo_metadata,
    )

    st.info(
        f"""
        Available photos: {photo_list.shape[0]}

        Unique roof ids: {photos_metadata.roof_id.unique().shape[0]}
        """
    )

    return chosen_folder, photos_metadata, photo_list
