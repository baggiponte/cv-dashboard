"""
String paths to the core dropbox data folders. Paths are collected here to ensure
that they are consistent across all the functions in the package.
"""

__all__ = [
    "DROPBOX_METADATA_PATH",
    "DROPBOX_HYPERPARAMETERS_PATH",
    "DROPBOX_RAW_PHOTOS_ROOT",
]

DROPBOX_METADATA_PATH = "/k2/metadata/transformed_data"
DROPBOX_HYPERPARAMETERS_PATH = "/k2/metadata/hyperparameters"
DROPBOX_RAW_PHOTOS_ROOT = "/k2/raw_photos"
