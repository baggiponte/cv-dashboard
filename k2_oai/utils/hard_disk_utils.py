import os


from k2_oai.utils.dropbox_io_utils import upload_file_to_path
from k2_oai.utils.dropbox_io_utils import *


def upload_hard_disk_data(dbx):
    root_hd_path = "/Volumes/Elements/Big_Size_Images_2"
    photos_filenames_list = os.listdir(root_hd_path)
    for photo_filename in photos_filenames_list[5000:10000]:
        if "Thumbs" not in photo_filename:
            print(photo_filename)
            upload_file_to_path(
                dbx,
                os.path.join(root_hd_path, photo_filename),
                "/k2/raw_photos/large_photos-5K_10K-api_upload/{}".format(photo_filename)
            )


dbx = dropbox_connect()
upload_hard_disk_data(dbx)
