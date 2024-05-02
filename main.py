import cv2
import streamlit as st
import helper_functions as hf
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import time

TEST = False

def start_blurring(video_file):
    # start timer using time module
    start = time.time()

    # If a video file is uploaded, complete the blurring process
    # this should be modularized once the algorithm is better refined
    if TEST == False:
        # display a progress bar
        st.write("Processing video...")
        my_bar = st.progress(0)

        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        # Load the video
        video = cv2.VideoCapture(temp_path)
        assert video.isOpened(), "Error reading video file"

        my_bar.progress(5)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to save the processed video
        output_path = "blurred_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Blur ratio
        blur_ratio = 50

        my_bar.progress(10)

        # set up the model
        model = YOLO("runs/detect/train10_50epochs/weights/best.pt")
        names = model.names

        my_bar.progress(15)

        
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #get frames at which to increase progress bar by 10%
        frames = [int(num_frames/10)*i for i in range(1, 10)]
        current_frame = 0

        # Process each frame of the video
        while video.isOpened():
            current_frame += 1
            if current_frame in frames:
                frame_percent = 10*frames.index(current_frame) + 15
                my_bar.progress(frame_percent)
                print("progress: ", frame_percent, "%")
            ret, frame = video.read()

            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.predict(frame, show=False, save_txt=True, save_conf=True, conf=0.7)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(frame, line_width=2, example=names)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                    frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj

            output_video.write(frame)

        video.release()
        output_video.release()

        my_bar.progress(100)
        end = time.time()
        #remove progress bar
        my_bar.empty()

        # Display the blurred video
        st.subheader("Blurred Video")
        st.video(output_path)
        st.text("Time taken to process video: {:.2f} seconds".format(end - start))

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

def start_page():
    st.title("Hide Brands in Your Video")
    st.markdown(
        """
        This app allows you to upload a video file and have any brands in our database be automatically blurred. 
        """
    )
    video_file = st.file_uploader("Upload a video file", type=["mp4"])
    if video_file and st.button("Start Blurring!"):
        start_blurring(video_file)

start_page()