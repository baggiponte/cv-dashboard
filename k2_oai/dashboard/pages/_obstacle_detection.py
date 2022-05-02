import matplotlib.pyplot as plt
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)


def obstacle_detection_page():

    # +-------------------------------------+
    # | Adjust title and sidebar, load data |
    # +-------------------------------------+

    st.title(":house_with_garden: Obstacle Detection Dashboard")

    with st.sidebar:
        st.subheader("Data Source")
        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=[
                "small_photos-api_upload",
            ]
            + [f"large_photos-{i}K_{i+5}K-api_upload" for i in range(0, 10, 5)],
        )

        photos_metadata, dbx_photo_list = utils.dbx_get_photos_and_metadata(
            chosen_folder
        )

        st.info(f"Available photos: {dbx_photo_list.shape[0]}")

        st.markdown("---")

    with st.expander("Click to see the list of photos"):
        st.dataframe(dbx_photo_list)

    with st.expander("Click to see the photos' metadata"):
        st.dataframe(photos_metadata)

    # +---------------------------------------+
    # | Choose roof id for obstacle detection |
    # +---------------------------------------+

    with st.sidebar:
        st.subheader("Target Roof")
        chosen_roof_id = int(
            st.selectbox(
                "Select the roof identifier: ",
                options=sorted(photos_metadata.roof_id.unique()),
            )
        )
        st.markdown("---")

    k2_labelled_image, bgr_roof, greyscale_roof = utils.crop_roofs_from_id(
        chosen_roof_id, photos_metadata, chosen_folder
    )

    # +-----------------------------+
    # | Obstacle Detection Pipeline |
    # +-----------------------------+

    st.subheader(f"Target roof's identifier: `{chosen_roof_id}`")

    with st.sidebar:

        st.subheader("Model Hyperparameters:")

        chosen_sigma = st.slider(
            "Insert sigma for filtering (positive, odd integer):",
            min_value=1,
            max_value=151,
            step=2,
        )

        chosen_filtering_method = st.radio(
            "Choose filtering method:", options=("Bilateral", "Gaussian")
        )

        chosen_binarisation_technique = st.radio(
            "Select the desired binarisation technique",
            ("Simple", "Adaptive", "Composite"),
        )

        if chosen_binarisation_technique == "Adaptive":
            chosen_blocksize = st.text_input(
                "Size of the pixel neighbourhood (positive, odd integer):", value=21
            )
        elif chosen_binarisation_technique == "Composite":
            chosen_tolerance = st.text_input(
                """
                Insert the desired tolerance for composite binarisation.
                If 'auto', tolerance will be deduced from the histogram's variance.
                """,
                value=10,
            )

        chosen_drawing_technique = st.radio(
            "Select the desired drawing technique", ("Bounding Box", "Bounding Polygon")
        )

    filtered_gs_roof = filtering_step(
        greyscale_roof, chosen_sigma, chosen_filtering_method.lower()
    )

    if chosen_binarisation_technique == "Simple":
        binarized_gs_roof = binarization_step(filtered_gs_roof, method="s")
    elif chosen_binarisation_technique == "Adaptive":
        binarized_gs_roof = binarization_step(
            filtered_gs_roof, method="a", adaptive_kernel_size=int(chosen_blocksize)
        )
    else:
        binarized_gs_roof = binarization_step(
            filtered_gs_roof, method="c", composite_tolerance=int(chosen_tolerance)
        )

    blurred_gs_roof = morphological_opening_step(binarized_gs_roof)

    boundary_type = "box" if chosen_drawing_technique == "Bounding Box" else "polygon"

    obstacles_blobs, roof_with_bboxes, obstacles_coordinates = detect_obstacles(
        blurred_roof=blurred_gs_roof,
        source_image=greyscale_roof,
        box_or_polygon=boundary_type,
        min_area="auto",
    )

    # +-------------------------+
    # | Roof & Color Histograms |
    # +-------------------------+

    st_roof_widgets = st.columns((1, 1))

    # original roof
    # -------------
    st_roof_widgets[0].image(
        k2_labelled_image,
        use_column_width=True,
        channels="BGRA",
        caption="Original image with database labels",
    )

    # RGB color histogram
    # -------------------
    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(
        bgr_roof[:, :, 0].flatten(), bins=50, edgecolor="blue", alpha=0.5
    )
    n, bins, patches = ax.hist(
        bgr_roof[:, :, 1].flatten(), bins=50, edgecolor="green", alpha=0.5
    )
    n, bins, patches = ax.hist(
        bgr_roof[:, :, 2].flatten(), bins=50, edgecolor="red", alpha=0.5
    )
    ax.set_title("Cropped Roof RGB Histogram")
    ax.set_xlim(0, 255)
    st_roof_widgets[1].pyplot(fig, use_column_width=True)

    # greyscale histogram
    # -------------------
    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(
        filtered_gs_roof.flatten(), bins=range(256), edgecolor="black", alpha=0.9
    )
    ax.set_title("Roof Greyscale Histogram After Filtering")
    ax.set_xlim(0, 255)
    st_roof_widgets[1].pyplot(fig, use_column_width=True)

    # +--------------------+
    # | Plot Model Results |
    # +--------------------+

    st.subheader("Obstacle Detection")

    st_results_widgets = st.columns((1, 1))

    st_results_widgets[0].image(
        bgr_roof,
        use_column_width=True,
        channels="BGRA",
        caption="Cropped Roof (RGB) with Database Labels",
    )

    st_results_widgets[0].image(
        filtered_gs_roof,
        use_column_width=True,
        caption="Cropped Roof (Greyscale) After Filtering",
    )

    st_results_widgets[1].image(
        (obstacles_blobs * 60) % 256,
        use_column_width=True,
        caption="Auto Obstacle Blobs (Greyscale)",
    )
    st_results_widgets[1].image(
        roof_with_bboxes,
        use_column_width=True,
        caption=f"Auto Labelled {chosen_drawing_technique}",
    )
