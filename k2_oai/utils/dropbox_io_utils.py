import os

import streamlit as st

import dropbox
from dropbox.exceptions import AuthError

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
EMAIL = os.environ.get("DROPBOX_USER_MAIL")
APP_KEY = os.environ.get("APP_KEY")
APP_SECRET = os.environ.get("APP_SECRET")


def dropbox_connect_oauth2_cmd():

    oauth_flow_dbx_obj = dropbox.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)

    authorize_url = oauth_flow_dbx_obj.start()

    print('1. Go to: ' + authorize_url)
    print('2. Click "Allow" (you might have to log in first)')
    print('3. Copy the authorization code.')
    auth_code = input("4. Enter the authorization code here: ")

    try:
        oauth_flow_dbx_obj = oauth_flow_dbx_obj.finish(auth_code)
        return oauth_flow_dbx_obj
    except:
        st.error("Invalid authorization code!")
        return None


def dropbox_connect_oauth2_streamlit():

    oauth_flow_dbx_obj = dropbox.DropboxOAuth2FlowNoRedirect(
        APP_KEY, APP_SECRET, token_access_type='offline',
    )

    authorize_url = oauth_flow_dbx_obj.start()

    _placeholders_list = list()
    for i in range(4):
        _placeholders_list.append(st.empty())

    with _placeholders_list[0]:
        st.write('1. Go to: ' + authorize_url)
    with _placeholders_list[1]:
        st.write('2. Click "Allow" (you might have to log in first)')
    with _placeholders_list[2]:
        st.write('3. Copy the authorization code.')
    with _placeholders_list[3]:
        auth_code = st.text_input("4. Enter the authorization code here: ")

    try:
        oauth_flow_dbx_obj = oauth_flow_dbx_obj.finish(auth_code)
        print(oauth_flow_dbx_obj.expires_at)
        return _placeholders_list, oauth_flow_dbx_obj
    except:
        st.error("Invalid authorization code!")
        return _placeholders_list, None


def dropbox_connect(dbx_access_token, dbx_refresh_token):

    """Create a connection to Dropbox."""

    try:

        dbx_team = dropbox.dropbox_client.DropboxTeam(
            oauth2_access_token=dbx_access_token, oauth2_refresh_token=dbx_refresh_token,
            app_key=APP_KEY, app_secret=APP_SECRET
        )
        #print(dbx_team.team_get_info())

        team_member_info = dbx_team.team_members_get_info(
            [dropbox.team.UserSelectorArg("email", EMAIL)]
        )[0].get_member_info()
        #print(team_member_info.profile)
        #print(team_member_info.profile.team_member_id)

        team_member_id = team_member_info.profile.team_member_id
        dbx = dropbox.dropbox_client.DropboxTeam(
            oauth2_access_token=dbx_access_token, oauth2_refresh_token=dbx_refresh_token,
            app_key=APP_KEY, app_secret=APP_SECRET
        ).with_path_root(dropbox.common.PathRoot.namespace_id(DROPBOX_NAMESPACE_ID)).as_user(
            team_member_id=team_member_id
        )
        #print(dbx.users_get_current_account())

        return dbx

    except AuthError as e:

        print('Error connecting to Dropbox with access token: ' + str(e))

        return None


def get_dropbox_list_files_df(dbx, path):

    """
    Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
    """

    if True:

        files_list = []

        list_folder_results = dbx.files_list_folder(path)

        while list_folder_results.has_more:
            files = list_folder_results.entries
            #print(files)
            for file in files:
                #print(file.path_display)
                if isinstance(file, dropbox.files.FolderMetadata):
                    metadata = {
                        'item_type': "folder",
                        'item_dropbox_id': file.id,
                        'item_name': file.name,
                        'item_abs_path': file.path_display,
                        #'client_modified': file.client_modified,
                        #'server_modified': file.server_modified
                    }
                    #print(getattr(metadata, "id"))
                    files_list.append(metadata)
                elif isinstance(file, dropbox.files.FileMetadata):
                    metadata = {
                        'item_type': "file",
                        'item_dropbox_id': file.id,
                        'item_name': file.name,
                        'item_abs_path': file.path_display,
                        #'client_modified': file.client_modified,
                        #'server_modified': file.server_modified
                    }
                    #print(getattr(metadata, "id"))
                    files_list.append(metadata)
            list_folder_results = dbx.files_list_folder_continue(list_folder_results.cursor)

        df = pd.DataFrame.from_records(files_list)
        return df

    else:
        print('Error getting list of files from Dropbox: ' + str(e))


def upload_file_to_path(dbx, file_path_from, file_path_to):
    with open(file_path_from, 'rb') as f:
        dbx.files_upload(f.read(), file_path_to)
