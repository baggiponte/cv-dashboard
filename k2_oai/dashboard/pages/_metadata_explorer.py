import altair as alt
import streamlit as st
from vega_datasets import data

from k2_oai.dashboard import utils


def metadata_explorer_page():

    st.title(":bar_chart: Metadata Explorer")
    st.write("Explore the satellite photos metadata.")

    # +-----------+
    # | Load Data |
    # +-----------+

    metadata = utils.dbx_get_metadata()

    # +---------------+
    # | Zoom Levels   |
    # +---------------+

    st.markdown(
        """
    ## Photos Distribution by Zoom Level

    Each increase in zoom level is twice as large in both the x and y directions.
    Therefore, each higher zoom level results in a resolution four times higher than the
    preceding level. For example, at zoom level 0 the map consists of one single 256x256
    pixel tile. At zoom level 1, the map consists of four 256x256 pixel tiles, resulting
    in a pixel space from 512x512.

    Furthermore, the zoom level range depends on the world area you are looking at. In
    other words, in some parts of the world, zoom is available up to only a certain
    level.
    """
    )

    zoom_levels = metadata.groupby("zoom").size()
    st.bar_chart(zoom_levels)

    # +----------------+
    # | World Map      |
    # +----------------+

    st.subheader("Zoom Level Distribution by Country")

    # TODO: just split the datapoints by continent and do a histogram.

    geo_metadata = metadata.dropna(subset=["center_lng", "center_lat"]).rename(
        columns={"center_lng": "lon", "center_lat": "lat"}
    )

    world = alt.topo_feature(data.world_110m.url, "countries")

    background = (
        alt.Chart(world)
        .mark_geoshape(fill="lightgray", stroke="white")
        .properties(width=1000, height=500)
        .project("mercator")
    )

    points = (
        alt.Chart(geo_metadata)
        .mark_circle()
        .encode(
            longitude="lon:Q",
            latitude="lat:Q",
            color="zoom:N",
            # size="zoom:N",
            tooltip=["zoom"],
        )
    )

    fig = background + points

    st.altair_chart(fig, use_container_width=True)
