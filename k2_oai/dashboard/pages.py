import os

import matplotlib.pyplot as plt
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)
from k2_oai.utils import draw_boundaries, rotate_and_crop_roof


def obstacle_detection_page():

    chosen_folder = None
    try:
        chosen_folder = st.selectbox(
            "Select photos folder name: ",
            options=[
                "small_photos-api_upload",
            ]
            + [f"large_photos-{i}K_{i+5}K-api_upload" for i in range(0, 10, 5)],
        )
    except:  # noqa: E722
        st.error("Please insert a valid folder")

    metadata = utils.get_photos_metadata()
    dbx_photos_list = utils.get_photos_list(folder_name=chosen_folder)

    if os.path.exists("join-roofs_images_obstacles.csv"):
        os.remove("join-roofs_images_obstacles.csv")

    if os.path.exists("join-roofs_images_obstacles.parquet"):
        os.remove("join-roofs_images_obstacles.parquet")

    st.write("Available photos: ", dbx_photos_list.shape[0])
    with st.expander("Click to see the list of photos on remote folder"):
        st.dataframe(dbx_photos_list)

    metadata = metadata[metadata.imageURL.isin(dbx_photos_list.item_name.values)]
    with st.expander("Click to see the metadata of the available photos"):
        st.dataframe(metadata)

    roof_id = None
    try:
        roof_id = int(
            st.selectbox("Select roof_id: ", options=sorted(metadata.roof_id.unique()))
        )
    except:  # noqa: E722
        st.error("Please insert a valid roof_id")

    st.write("Roof obstacles metadata as recorded in the database:")

    metadata_photo_obstacles = metadata.loc[metadata.roof_id == roof_id]
    st.dataframe(metadata_photo_obstacles)

    pixel_coord_roof = metadata.loc[
        metadata.roof_id == roof_id, "pixelCoordinates_roof"
    ].iloc[0]

    pixel_coord_obs = [
        coord
        for coord in metadata.loc[
            metadata.roof_id == roof_id, "pixelCoordinates_obstacle"
        ].values
    ]

    st.write("Number of obstacles in the database: ", len(pixel_coord_obs))

    photo_name = metadata.loc[lambda df: df["roof_id"] == roof_id, "imageURL"].values[0]

    bgr_image, greyscale_image = utils.load_photo_from_dbx(chosen_folder, photo_name)
    if os.path.exists(photo_name):
        os.remove(photo_name)

    k2_labelled_image = draw_boundaries(bgr_image, pixel_coord_roof, pixel_coord_obs)
    greyscale_roof = rotate_and_crop_roof(greyscale_image, pixel_coord_roof)
    bgr_roof = rotate_and_crop_roof(k2_labelled_image, pixel_coord_roof)

    st_cols = st.columns((1, 1))
    st_cols[0].image(
        k2_labelled_image,
        use_column_width=True,
        channels="BGRA",
        caption="Original image with database labels",
    )
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
    st_cols[1].pyplot(fig, use_column_width=True)

    st_cols_widgets = st.columns((1, 1, 1))

    chosen_sigma = st_cols_widgets[0].text_input(
        "Insert sigma for Gaussian filtering (positive, odd integer):", value=1
    )
    # add input for filtering method?
    # chosen_filtering_method = st_cols_widgets[0].radio(
    #     "Choose filtering method:", options=("Bilateral", "Gaussian")
    # )
    filtered_gs_roof = filtering_step(greyscale_roof, int(chosen_sigma), "b")

    chosen_binarisation_technique = st_cols_widgets[0].radio(
        "Select the desired binarisation technique",
        ("Simple", "Adaptive", "Composite"),
    )

    if chosen_binarisation_technique == "Simple":
        binarized_gs_roof = binarization_step(filtered_gs_roof, method="s")
    elif chosen_binarisation_technique == "Adaptive":
        chosen_blocksize = st_cols_widgets[0].text_input(
            """
            Choose the kernel size (size of pixel neighborhood, positive, odd integer):
            """,
            value=21,
        )
        binarized_gs_roof = binarization_step(
            filtered_gs_roof, method="a", adaptive_kernel_size=int(chosen_blocksize)
        )
    elif chosen_binarisation_technique == "Composite":
        chosen_tolerance = st_cols_widgets[0].text_input(
            """
            Insert the desired tolerance for composite binarisation.
            If 'auto', tolerance will be deduced from the histogram's variance.
            """,
            value=10,
        )
        binarized_gs_roof = binarization_step(
            filtered_gs_roof, method="c", composite_tolerance=int(chosen_tolerance)
        )
    # why is this needed?
    else:
        binarized_gs_roof = binarization_step(filtered_gs_roof, method="s")

    blurred_gs_roof = morphological_opening_step(binarized_gs_roof)

    chosen_drawing_technique = st_cols_widgets[0].radio(
        "Select the desired drawing technique", ("Bounding Box", "Bounding Polygon")
    )
    if chosen_drawing_technique == "Bounding Box":
        boundary_type = "box"
    else:
        boundary_type = "polygon"

    obstacles_blobs, roof_with_bboxes, obstacles_coordinates = detect_obstacles(
        blurred_roof=blurred_gs_roof,
        source_image=greyscale_roof,
        box_or_polygon=boundary_type,
        min_area="auto",
    )

    # im_draw, im_result, im_error, im_rel_area_error = surface_absolute_error(
    #     binarized_gs_roof,
    #     pixel_coord_roof,
    #     pixel_coord_obs,
    #     obstacles_coordinates
    # )

    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(
        filtered_gs_roof.flatten(), bins=range(256), edgecolor="black", alpha=0.9
    )
    ax.set_title("Cropped Roof Greyscale Histogram")
    ax.set_xlim(0, 255)
    st_cols[1].pyplot(fig, use_column_width=True)

    # st_cols = st.columns((1, 1))

    st_cols_widgets[1].image(
        bgr_roof,
        use_column_width=True,
        channels="BGRA",
        caption="Cropped Roof (RGB) with Database Labels",
    )
    # st_cols[1].image(greyscale_roof, use_column_width=True, caption="Cropped Roof GS")
    st_cols_widgets[2].image(
        filtered_gs_roof,
        use_column_width=True,
        caption="Cropped Roof (Greyscale) After Filtering",
    )

    # st_cols = st.columns((1, 1, 5))

    st_cols_widgets[1].image(
        (obstacles_blobs * 60) % 256,
        use_column_width=True,
        caption="Auto Obstacle Blobs (Greyscale)",
    )
    st_cols_widgets[2].image(
        roof_with_bboxes,
        use_column_width=True,
        caption=f"Auto Labelled {chosen_drawing_technique}",
    )
    # st_cols[2].image(
    #   im_error, use_column_width=True, caption="Auto labels VS DB labels"
    # )
