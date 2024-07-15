import os
from pathlib import Path
# print(os.getcwd(), str(os.getcwd()).split("\\")[-1])
# if str(os.getcwd()).split("\\")[-1]=="MOTTracker":
#     print(os.getcwd())

import shutil
from utils.general import cv2
import detector_tracker
import streamlit as st

from PIL import Image
from pathlib import Path
import base64

if os.path.exists('runs'):
    shutil.rmtree('runs')


def save_uploaded_file(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved file: {uploadedfile.name}")
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.markdown("<p style='text-align: right; color: white;'> "+img_to_html('kpmg.png')+"</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'> "+img_to_html('national_emblem_resized.png')+"</p>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'>Computer Vision</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'>Multi Object Detector and Tracker Demo</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>KPMG DEMO</h3>", unsafe_allow_html=True)
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")
st.write("\n\n\n\n\n")

st.markdown("<h2 style='text-align: center; color: blue;'>Yolov9 and Deepsort</h2>", unsafe_allow_html=True)
#st.markdown("<h4 style='text-align: center; color: black;'>Acne and Actinic Skin Disease Classification</h4>", unsafe_allow_html=True)
st.title("Upload and Save Video or Image File")

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






