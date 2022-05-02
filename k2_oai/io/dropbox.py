import os

import dropbox
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from dropbox.exceptions import AuthError

load_dotenv()

DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
DROPBOX_USER_EMAIL = os.environ.get("DROPBOX_USER_MAIL")
DROPBOX_APP_KEY = os.environ.get("APP_KEY")
DROPBOX_APP_SECRET = os.environ.get("APP_SECRET")
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")


def dropbox_oauth2_connect(
    dbx_app_key=DROPBOX_APP_KEY, dbx_app_secret=DROPBOX_APP_SECRET
):

    dropbox_oauth2_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dbx_app_key, dbx_app_secret
    )

    authorization_url = dropbox_oauth2_flow.start()

    print(
        f"""
        1. Go to: {authorization_url}
        2. Click "Allow" (you might have to log in first)
        3. Copy the authorization code.
        """
    )
    authorization_code = input("4. Enter the authorization code here: ")

    try:
        return dropbox_oauth2_flow.finish(authorization_code)
    except:  # noqa: E722
        print("Invalid authorization code!")
        return None


def st_dropbox_oauth2_connect(
    dbx_app_key=DROPBOX_APP_KEY, dbx_app_secret=DROPBOX_APP_SECRET
):
    dropbox_oauth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dbx_app_key,
        dbx_app_secret,
        token_access_type="offline",
    )

    authorization_url = dropbox_oauth_flow.start()

    placeholder = st.empty()

    with placeholder.container():
        st.title(":key: Dropbox Authentication")
        st.markdown(f"1. Go to [this url]({authorization_url}).")
        st.write('2. Click "Allow" (you might have to log in first).')
        st.write("3. Copy the authorization code.")
        st.write("4. Enter the authorization code here: ")
        authorization_code = st.text_input("")

    try:
        complete_oauth_flow = dropbox_oauth_flow.finish(authorization_code)
        return placeholder, complete_oauth_flow
    except:  # noqa: E722
        if authorization_code is not None and len(authorization_code):
            st.error("Invalid authorization code!")
        return placeholder, None


def _get_team_member_id(dbx_team_app, dbx_user_email):
    return (
        dbx_team_app.team_members_get_info(
            [dropbox.team.UserSelectorArg("email", dbx_user_email)]
        )[0]
        .get_member_info()
        .profile.team_member_id
    )


def dropbox_connect_access_token_only(
    dbx_access_token,
    dbx_namespace_id=DROPBOX_NAMESPACE_ID,
    dbx_user_email=DROPBOX_USER_EMAIL,
):
    try:
        dbx_team = dropbox.DropboxTeam(dbx_access_token)

        dbx_team_member_id = _get_team_member_id(dbx_team, dbx_user_email)

        return dbx_team.with_path_root(
            dropbox.common.PathRoot.namespace_id(dbx_namespace_id)
        ).as_user(team_member_id=dbx_team_member_id)

    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
        return None


def dropbox_connect(
    dbx_access_token,
    dbx_refresh_token,
    dbx_app_key=DROPBOX_APP_KEY,
    dbx_app_secret=DROPBOX_APP_SECRET,
    dbx_user_email=DROPBOX_USER_EMAIL,
    dbx_namespace_id=DROPBOX_NAMESPACE_ID,
):
    """Create a connection to Dropbox."""

    try:

        dbx_team = dropbox.DropboxTeam(
            oauth2_access_token=dbx_access_token,
            oauth2_refresh_token=dbx_refresh_token,
            app_key=dbx_app_key,
            app_secret=dbx_app_secret,
        )

        team_member_id = _get_team_member_id(dbx_team, dbx_user_email)

        return dbx_team.with_path_root(
            dropbox.common.PathRoot.namespace_id(dbx_namespace_id)
        ).as_user(team_member_id=team_member_id)

    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
        return None


def _parse_dropbox_folder_content(folder_content):
    files_list = []

    files = folder_content.entries

    for file in files:
        if isinstance(file, dropbox.files.FolderMetadata):
            metadata = {
                "item_type": "folder",
                "item_dropbox_id": file.id,
                "item_name": file.name,
                "item_abs_path": file.path_display,
                # 'client_modified': file.client_modified,
                # 'server_modified': file.server_modified
            }
            files_list.append(metadata)
        elif isinstance(file, dropbox.files.FileMetadata):
            metadata = {
                "item_type": "file",
                "item_dropbox_id": file.id,
                "item_name": file.name,
                "item_abs_path": file.path_display,
                # 'client_modified': file.client_modified,
                # 'server_modified': file.server_modified
            }
            files_list.append(metadata)

    return files_list


def list_content_of(dbx_app, path):
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps
    directory.
    """

    files_list = []

    dbx_folder_contents = dbx_app.files_list_folder(path)
    files = dbx_folder_contents.entries

    while dbx_folder_contents.has_more or len(files):
        files = dbx_folder_contents.entries
        files_list += _parse_dropbox_folder_content(dbx_folder_contents)
        dbx_folder_contents = dbx_app.files_list_folder_continue(
            dbx_folder_contents.cursor
        )

    return pd.DataFrame.from_records(files_list)


def upload_file_to_dropbox(dbx_app, file_path_from, file_path_to):
    with open(file_path_from, "rb") as f:
        dbx_app.files_upload(f.read(), file_path_to)


def download_from_dropbox(dbx_app, file_path_from, file_path_to):
    dbx_app.files_download_to_file(file_path_from, file_path_to)
