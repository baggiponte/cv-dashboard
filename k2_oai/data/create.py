"""
Functions to manipulate the raw data.
"""

import geopandas

from k2_oai.data.load import dbx_load_metadata

__all__ = [
    "dbx_create_geo_metadata",
]


def dbx_create_geo_metadata(dropbox_app):
    metadata = (
        dbx_load_metadata(dropbox_app=dropbox_app)
        .dropna(subset=["center_lng", "center_lat"])
        .rename(columns={"center_lng": "lon", "center_lat": "lat"})
    )

    return geopandas.GeoDataFrame(
        metadata,
        geometry=geopandas.points_from_xy(metadata.lon, metadata.lat),
        crs=4326,
    )
