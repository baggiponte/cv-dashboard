import matplotlib.pyplot as plt

from k2_oai.image_segmentation import *

from k2_oai.utils.st_utils import *


def obstacle_detection_page(placeholders_list):

    for placeholder in placeholders_list:
        placeholder.empty()

    def plot_channel_histogram(im_in):
        return cv.calcHist(im_in, [0], None, [256], [0, 256])

    chosen_folder = None
    try:
        chosen_folder = st.selectbox(
            "Select photos folder name: ",
            options=[
                "small_photos-api_upload",
            ] + [
                "large_photos-{}K_{}K-api_upload".format(str(i), str(i+5))
                for i in range(0, 10, 5)
            ],
        )
    except:
        st.error("Please insert a valid folder")

    metadata_df = get_metadata_df()
    dropbox_list_files_df = get_photos_list(folder_name=chosen_folder)

    if os.path.exists("inner_join-roofs_images_obstacles.csv"):
        os.remove("inner_join-roofs_images_obstacles.csv")

    st.write(dropbox_list_files_df.shape)
    with st.expander("Click to see the list of photos paths on remote folder"):
        st.dataframe(dropbox_list_files_df)

    metadata_df = metadata_df[metadata_df.imageURL.isin(dropbox_list_files_df.item_name.values)]
    with st.expander("Click to see the metadata table for available photos"):
        st.dataframe(metadata_df)

    roof_id = None
    try:
        roof_id = int(st.selectbox("Select roof_id: ", options=sorted(metadata_df.roof_id.unique())))
    except:
        st.error("Please insert a valid roof_id")

    st.write("Roof obstacles metadata as recorded in DB:")

    metadata_photo_obstacles = metadata_df.loc[metadata_df.roof_id == roof_id]

    st.dataframe(metadata_photo_obstacles)

    pixel_coord_roof = metadata_df.loc[
        metadata_df.roof_id == roof_id,
        "pixelCoordinates_roof"
    ].iloc[0]

    pixel_coord_obs = [
        coord for coord in metadata_df.loc[
            metadata_df.roof_id == roof_id,
            "pixelCoordinates_obstacle"
        ].values
    ]

    st.write("Number of obstacles in DB:", len(pixel_coord_obs))

    test_photo_name = metadata_df.loc[lambda df: df["roof_id"] == roof_id, "imageURL"].values[0]

    im_bgr, im_gs = load_photo_from_remote(chosen_folder, test_photo_name)
    if os.path.exists(test_photo_name):
        os.remove(test_photo_name)

    im_k2labeled = draw_boundaries(
        im_bgr, pixel_coord_roof, pixel_coord_obs
    )
    im_gs_cropped = rotate_and_crop_roof(im_gs, pixel_coord_roof)
    im_bgr_cropped = rotate_and_crop_roof(im_k2labeled, pixel_coord_roof)

    st_cols = st.columns((1, 1))
    st_cols[0].image(im_k2labeled, use_column_width=True, channels="BGRA", caption="Original image with DB labels")
    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 0].flatten(), bins=50, edgecolor='blue', alpha=0.5)
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 1].flatten(), bins=50, edgecolor='green', alpha=0.5)
    n, bins, patches = ax.hist(im_bgr_cropped[:, :, 2].flatten(), bins=50, edgecolor='red', alpha=0.5)
    ax.set_title("RGB crop histogram")
    ax.set_xlim(0, 255)
    st_cols[1].pyplot(fig, use_column_width=True)

    st_cols_widgets = st.columns((1, 1, 1))

    chosen_sigma = st_cols_widgets[0].text_input(
        "Insert sigma for Gaussian filtering (odd int)",
        value=1
    )
    im_filtered = apply_filter(im_gs_cropped, int(chosen_sigma), "g")

    chosen_binarisation_technique = st_cols_widgets[0].radio(
        "Select the desired binarisation technique",
        ('Simple', 'Adaptive', 'Composite'),
    )

    if chosen_binarisation_technique == 'Simple':
        im_thresholded = apply_binarization(im_filtered, "s")
    elif chosen_binarisation_technique == 'Adaptive':
        chosen_blocksize = st_cols_widgets[0].text_input(
            """
            Insert the desired blocksize (size of pixel neighborhood, odd int) for adaptive binarisation. 
            """,
            value=21
        )
        im_thresholded = apply_binarization(im_filtered, "a", blocksize=int(chosen_blocksize))
    elif chosen_binarisation_technique == 'Composite':
        chosen_tolerance = st_cols_widgets[0].text_input(
            """
            Insert the desired tolerance for composite binarisation. 
            Type 'infer' to try deducing the right tolerance from the variance of GS histogram.
            """,
            value=10
        )
        if chosen_tolerance != "infer":
            im_thresholded = apply_binarization(im_filtered, "c", tol=float(chosen_tolerance))
        else:
            im_thresholded = apply_binarization(im_filtered, "c", tol=str(chosen_tolerance))
    else:
        im_thresholded = apply_binarization(im_filtered, "s")

    if im_gs_cropped.size > 10000:
        opening = 3
    else:
        opening = 1

    chosen_drawing_technique = st_cols_widgets[0].radio(
        "Select the desired drawing technique",
        ('Bounding Box', 'Bounding Polygon')
    )
    if chosen_drawing_technique == "Bounding Box":
        bbox_or_polygon = "bbox"
    elif chosen_drawing_technique == "Bounding Polygon":
        bbox_or_polygon = "polygon"

    im_segmented, im_bbox, rect_coord = image_segmentation(
        im_thresholded,
        im_gs_cropped,
        opening,
        min_area=int(np.max(im_thresholded.shape)/10),
        bbox_or_polygon=bbox_or_polygon
    )

    # im_draw, im_result, im_error, im_rel_area_error = surface_absolute_error(
    #     im_thresholded,
    #     pixel_coord_roof,
    #     pixel_coord_obs,
    #     rect_coord
    # )

    fig, ax = plt.subplots(figsize=(3, 1))
    n, bins, patches = ax.hist(im_filtered.flatten(), bins=range(256), edgecolor='black', alpha=0.9)
    ax.set_title("GS crop histogram")
    ax.set_xlim(0, 255)
    st_cols[1].pyplot(fig, use_column_width=True)

    #st_cols = st.columns((1, 1))

    st_cols_widgets[1].image(im_bgr_cropped, use_column_width=True, channels="BGRA", caption="Cropped Roof RGB with DB labels")
    #st_cols[1].image(im_gs_cropped, use_column_width=True, caption="Cropped Roof GS")
    st_cols_widgets[2].image(im_filtered, use_column_width=True, caption="Cropped Roof GS filtered")

    #st_cols = st.columns((1, 1, 5))

    st_cols_widgets[1].image((im_segmented * 60) % 256, use_column_width=True, caption="Auto GS Blobs")
    st_cols_widgets[2].image(im_bbox, use_column_width=True, caption="Auto labels {}".format(chosen_drawing_technique))
    #st_cols[2].image(im_error, use_column_width=True, caption="Auto labels VS DB labels")
