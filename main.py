import cv2
import streamlit as st
import helper_functions as hf

TEST = True

st.title("Hide Brands in Your Video")
st.markdown(
    """
    This app allows you to upload a video file and have any brands in our database be automatically blurred. 
    """
)
video_file = st.file_uploader("Upload a video file", type=["mp4"])

# If a video file is uploaded, complete the blurring process
# this should be modularized once the algorithm is better refined
if video_file is not None and TEST == False:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    # Load the video
    video = cv2.VideoCapture(temp_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the processed video
    output_path = "blurred_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame of the video
    while True:
        ret, frame = video.read()

        if not ret:
            break

        objects = hf.identify_objects(frame)
        cropped_objects = hf.crop_and_convert_objects(frame, objects)

        transformed_objects = []
        for obj in cropped_objects:
            transformed_obj = hf.transform_image(obj)
            transformed_objects.append(transformed_obj)

        blurred_frame = hf.match_and_blur_brands(frame, transformed_objects)
        output_video.write(blurred_frame)

    video.release()
    output_video.release()

    # Display the blurred video
    st.subheader("Blurred Video")
    st.video(output_path)

    st.download_button(
        label="Download Resulting Video",
        data=output_path,
        file_name="blurred_video.mp4",
        mime="video/mp4"
    )

elif video_file is not None and TEST:
    st.subheader("Blurred Video")
    st.video(video_file)
    st.download_button(
        label="Download Resulting Video",
        data=video_file,
        file_name="blurred_video.mp4",
        mime="video/mp4"
    )
    
