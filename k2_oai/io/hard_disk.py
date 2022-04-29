import os

from k2_oai.io.dropbox import upload_file_to_dropbox

_HARD_DISK_PATH = "/Volumes/Elements/Big_Size_Images_2"


def upload_hard_disk_data(dropbox_app, hard_disk_path = None):
    hd_path = _HARD_DISK_PATH if hard_disk_path is None else hard_disk_path
    photo_list = os.listdir(hd_path)
    for photo in photo_list[5000:10000]:
        if "Thumbs" not in photo:
            print(photo)
            upload_file_to_dropbox(
                dropbox_app,
                os.path.join(hd_path, photo),
                f"/k2/raw_photos/large_photos-5K_10K-api_upload/{photo}"
            )
