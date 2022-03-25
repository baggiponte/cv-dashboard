import os

import dropbox
from dropbox.exceptions import AuthError

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")
DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
EMAIL = os.environ.get("DROPBOX_USER_MAIL")


def dropbox_connect(email=EMAIL):

    """Create a connection to Dropbox."""

    try:
        dbx_team = dropbox.dropbox_client.DropboxTeam(
            DROPBOX_ACCESS_TOKEN
        )
        print(dbx_team.team_get_info())

        team_member_info = dbx_team.team_members_get_info(
            [dropbox.team.UserSelectorArg("email", email)]
        )[0].get_member_info()
        print(team_member_info.profile)
        print(team_member_info.profile.team_member_id)

        team_member_id = team_member_info.profile.team_member_id
        dbx = dropbox.dropbox_client.DropboxTeam(
            DROPBOX_ACCESS_TOKEN
        ).with_path_root(dropbox.common.PathRoot.namespace_id(DROPBOX_NAMESPACE_ID)).as_user(
            team_member_id=team_member_id
        )
        print(dbx.users_get_current_account())

    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))

    return dbx


def get_dropbox_list_files_df(dbx, path):

    """
    Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
    """

    if True:
        files = dbx.files_list_folder(path).entries
        #print(files)
        files_list = []
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

        df = pd.DataFrame.from_records(files_list)
        return df

    else:
        print('Error getting list of files from Dropbox: ' + str(e))

