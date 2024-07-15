import os
print(os.getcwd())
if str(os.getcwd()).split("MOTTracker")[-1]=="MOTTracker":
    os.chdir(os.path.join(os.getcwd(), "yolov9"))
print(os.getcwd())

import shutil
from utils.general import cv2
import detector_tracker
import streamlit as st

from PIL import Image
from pathlib import Path

if os.path.exists('runs'):
    shutil.rmtree('runs')


def save_uploaded_file(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved file: {uploadedfile.name}")

st.title("Upload and Save Video File")

# Upload the video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if str(uploaded_file.name).split(".")[-1] in ["mp4", "avi", "mov", "mkv"]:
        # Display video in the app
        st.video(uploaded_file)
    elif str(uploaded_file.name).split(".")[-1] in ["jpg", "jpeg", "png"]:
        st.image(uploaded_file)

    # Save the uploaded file
    save_uploaded_file(uploaded_file)
    st.write(uploaded_file.name)

    frame_info = detector_tracker.run(
        weights='yolov9-c.pt',
        source=uploaded_file.name,
        device='cpu')

    if str(uploaded_file.name).split(".")[-1] in ["jpg", "jpeg", "png"]:
        image = cv2.imread("bb"+uploaded_file.name)

        # Resize the image
        resized_image = cv2.resize(image, (400, 300))  # Resize to 400x300

        # Convert the image to RGB format (OpenCV loads images in BGR format)
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Convert the image to a format that Streamlit can display
        resized_image_pil = Image.fromarray(resized_image_rgb)

        # Display the image using Streamlit
        st.image(resized_image_pil, caption='Detected', use_column_width=True)
        st.text(frame_info)
        os.remove("bb" + uploaded_file.name)

    if str(uploaded_file.name).split(".")[-1] == 'mp4':
        video_detactor_path = Path("runs\detect\exp")
        #st.video(os.path.join(video_detactor_path, uploaded_file.name))
        # print("I am Here", frame_info)
        st.text(frame_info)
        st.success("Video processing complete")
        #st.video(os.path.join(video_detactor_path, uploaded_file.name))
        with open(os.path.join(video_detactor_path, uploaded_file.name), "rb") as file:
            btn = st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    os.remove(uploaded_file.name)






