import matplotlib.pyplot as plt
import streamlit as st

from k2_oai.dashboard import utils


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

        roof_ids = sorted(photos_metadata.roof_id.unique())

        chosen_roof_id = int(
            st.selectbox(
                "Select the roof identifier: ",
                options=roof_ids,
            )
        )
        st.markdown("---")

    k2_labelled_image, bgr_roof, greyscale_roof = utils.crop_roofs_from_roof_id(
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
            chosen_tolerance = None
        elif chosen_binarisation_technique == "Composite":
            chosen_tolerance = st.text_input(
                """
                Insert the desired tolerance for composite binarisation.
                If 'auto', tolerance will be deduced from the histogram's variance.
                """,
                value=10,
            )
            chosen_blocksize = None
        else:
            chosen_blocksize, chosen_tolerance = None, None

        chosen_drawing_technique = st.radio(
            "Select the desired drawing technique", ("Bounding Box", "Bounding Polygon")
        )

    (
        obstacle_blobs,
        roof_with_bboxes,
        obstacles_coordinates,
        filtered_gs_roof,
    ) = utils.obstacle_detection_pipeline(
        greyscale_roof=greyscale_roof,
        sigma=chosen_sigma,
        filtering_method=chosen_filtering_method,
        binarization_method=chosen_binarisation_technique,
        blocksize=chosen_blocksize,
        tolerance=chosen_tolerance,
        boundary_type=chosen_drawing_technique,
        return_filtered_roof=True,
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

    st.subheader("Obstacle Detection, Visualized")

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
        (obstacle_blobs * 60) % 256,
        use_column_width=True,
        caption="Auto Obstacle Blobs (Greyscale)",
    )
    st_results_widgets[1].image(
        roof_with_bboxes,
        use_column_width=True,
        caption=f"Auto Labelled {chosen_drawing_technique}",
    )
