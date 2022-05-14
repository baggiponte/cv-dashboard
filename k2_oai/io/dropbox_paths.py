"""
String paths to the core dropbox data folders. Paths are collected here to ensure
that they are consistent across all the functions in the package.
"""

__all__ = [
    "DROPBOX_DESIGN_MATRIX_PATH",
    "DROPBOX_RAW_PHOTOS_ROOT",
    "DROPBOX_METADATA_PATH",
    "DROPBOX_HYPERPARAMETERS_PATH",
    "DROPBOX_ANNOTATIONS_PATH",
]

DROPBOX_METADATA_PATH = "/k2/metadata/transformed_data"
DROPBOX_HYPERPARAMETERS_PATH = "/k2/metadata/hyperparameters"
DROPBOX_ANNOTATIONS_PATH = "/k2/metadata/label_annotations"
DROPBOX_RAW_PHOTOS_ROOT = "/k2/raw_photos"
DROPBOX_DESIGN_MATRIX_PATH = "/k2/design_matrix"
