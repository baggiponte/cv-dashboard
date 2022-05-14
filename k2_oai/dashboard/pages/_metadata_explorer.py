"""
Dashboard mode to view and explore roof distribution in the world, as well as the
satellite photos' zoom levels.
"""

import altair as alt
import streamlit as st

from k2_oai.dashboard import utils

__all__ = ["metadata_explorer_page"]


def metadata_explorer_page():

    st.title(":bar_chart: Metadata Explorer")

    # +-----------+
    # | Load Data |
    # +-----------+

    metadata = utils.st_load_geo_metadata()
    # earth = utils.st_load_earth()

    # +---------------+
    # | Zoom Levels   |
    # +---------------+

    with st.sidebar:
        st.subheader("Photos Distribution by Zoom Level")
        st.write(
            """
        Each increase in zoom level is twice as large in both the x and y directions.
        Therefore, each higher zoom level results in a resolution four times higher than
        the preceding level. For example, at zoom level 0 the map consists of one single
        256x256 pixel tile. At zoom level 1, the map consists of four 256x256 pixel
        tiles, resulting in a pixel space from 512x512.

        Furthermore, the zoom level range depends on the world area you are looking at.
        In other words, in some parts of the world, zoom is available up to only a
        certain level.
        """
        )

    zoom_levels = metadata.groupby("zoom").size()
    st.bar_chart(zoom_levels)

    # +-------------------------+
    # | Zoom level by continent |
    # +-------------------------+

    st.subheader("Zoom Level Distribution by Continent")

    zoom_levels_by_continent = (
        metadata.groupby(["continent", "zoom"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .astype({"zoom": "int8"})
        .set_index("continent")
    )

    st_plot, st_selector = st.columns((3, 1))

    st_selector.selectbox(
        "Select a continent",
        options=(zoom_levels_by_continent.index.unique()),
        key="continent_selector",
    )

    fig = (
        alt.Chart(
            zoom_levels_by_continent.loc[
                lambda df: df.index == st.session_state["continent_selector"]
            ]
        )
        .mark_bar()
        .encode(x="zoom:N", y="count:Q")
    )

    st_plot.altair_chart(fig, use_container_width=True)

    # +----------------+
    # | World Map      |
    # +----------------+

    st.subheader("Zoom Level Distribution by Country")

    zoom_levels_by_country = (
        metadata.groupby(["name", "zoom"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .astype({"zoom": "int8"})
        .set_index("name")
    )

    st_plot, st_selector = st.columns((3, 1))

    st_selector.selectbox(
        "Select a continent",
        options=(zoom_levels_by_country.index.unique()),
        key="country_selector",
    )

    fig = (
        alt.Chart(
            zoom_levels_by_country.loc[
                lambda df: df.index == st.session_state["country_selector"]
            ]
        )
        .mark_bar()
        .encode(x="zoom:N", y="count:Q")
    )

    st_plot.altair_chart(fig, use_container_width=True)

    #
    # background = (
    #     alt.Chart(world)
    #     .mark_geoshape(fill="lightgray", stroke="white")
    #     .properties(width=1000, height=500)
    #     .project("mercator")
    # )
    #
    # points = (
    #     alt.Chart(metadata)
    #     .mark_circle()
    #     .encode(
    #         longitude="lon:Q",
    #         latitude="lat:Q",
    #         color="zoom:N",
    #         # size="zoom:N",
    #         tooltip=["zoom"],
    #     )
    # )
    #
    # fig = background + points
    #
    # st.altair_chart(fig, use_container_width=True)
