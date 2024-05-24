import sys
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the module path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'data', '5_Image_Processing')
if module_path not in sys.path:
    sys.path.append(module_path)

import Chapter03 as c3  # type: ignore
import Chapter04 as c4  # type: ignore
import Chapter05 as c5  # type: ignore
import Chapter09 as c9  # type: ignore
import tempfile

st.set_page_config(page_title="Xử lý ảnh", page_icon="🖼️")

st.header("Xử lý ảnh")

filter_functions = {
    "Chapter 03": {
        "Negative": c3.Negative,
        "NegativeColor": c3.NegativeColor,
        "Logarit": c3.Logarit,
        "LogaritColor": c3.LogaritColor,
        "Power": c3.Power,
        "PowerColor": c3.PowerColor,
        "PiecewiseLinear": c3.PiecewiseLinear,
        "PiecewiseLinearColor": c3.PiecewiseLinearColor,
        "Histogram": c3.Histogram,
        "HistEqual": c3.HistEqual,
        "HistEqualColor": c3.HistEqualColor,
        "LocalHist": c3.LocalHist,
        "HistStat": c3.HistStat,
        "BoxFilter": c3.MyFilter2D,
        "Threshold": c3.Threshold,
        "MedianFilter": c3.MyMedian,
        "Sharpen": c3.Sharpen,
        "Gradient": c3.Gradient,
    },
    "Chapter 04": {
        "Spectrum": c4.Spectrum,
        "FrequencyFilter": c4.FrequencyFilter,
        "RemoveMoire": c4.RemoveMoire
    },
    "Chapter 05": {
        "CreateMotionfilter": c5.CreateMotionfilter,
        "CreateMotionNoise": c5.CreateMotionNoise,
        "CreateInverseMotionfilter": c5.CreateInverseMotionfilter,
        "DenoiseMotion": c5.DenoiseMotion,
    },
    "Chapter 09": {
        "Erosion": c9.Erosion,
        "Dilation": c9.Dilation,
        "OpeningClosing": c9.OpeningClosing,
        "Boundary": c9.Boundary,
        "HoleFill": c9.HoleFill,
        "MyConnectedComponent": c9.MyConnectedComponent,
        "ConnectedComponent": c9.ConnectedComponent,
        "CountRice": c9.CountRice
    }
}

def apply_filter(img_array, chapter, filter_type):
    if filter_type in filter_functions[chapter]:
        if filter_type == "DrawNotchRejectFilter":
            return filter_functions[chapter][filter_type]()
        else:
            return filter_functions[chapter][filter_type](img_array)
    else:
        st.warning("Selected processing not available.")
        return img_array

def create_dropdown_menu(chapter, filter_functions):
    selected_filter = st.selectbox(f"Choose a processing {chapter}", [f"Select Processing {chapter}"] + list(filter_functions.keys()))
    return selected_filter

def detect_image_type(image):
    average_color = np.mean(image, axis=(0, 1))
    if type(average_color) == np.ndarray:
        # nếu giá trị trung bình có nhiều giá trị
        if average_color[0] == average_color[1] == average_color[2]:
            # print("Ảnh xám")
            return "gray"
        else:
            # print("Ảnh màu")
            return "color"
    else:
        # print("Ảnh xám")
        return "gray"

# File uploader
uploaded_file_01 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "bmp"])
if uploaded_file_01 is not None:
    image = Image.open(uploaded_file_01)
    img_type = detect_image_type(image)
    col1, col2 = st.columns(2)
    new_image_uploaded = False

    # Initialize selected_chapter outside the block
    selected_chapter = "Select a chapter"
    selected_filter = "Select Processing Chapter 03"

    with col1:
        if uploaded_file_01 is not None:
            selected_chapter = "Select a chapter"
            selected_filter = "Select Processing Chapter 03"
            new_image_uploaded = True

            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name, format='PNG')  # Save the uploaded PIL image to the temporary file

            # Read the image using OpenCV
            if img_type == "gray":
                img_array = cv2.imread(temp_file.name, cv2.IMREAD_GRAYSCALE)
            else:
                img_array = cv2.imread(temp_file.name, cv2.IMREAD_COLOR)

        if new_image_uploaded:
            selected_chapter = st.selectbox("Choose a chapter", ['Select a chapter', 'Chapter 03', 'Chapter 04', 'Chapter 05', 'Chapter 09'])
            st.image(image, caption="Uploaded Image", use_column_width=True)
            new_image_uploaded = False  # Reset the flag after processing

    with col2:
        if selected_chapter != "Select a chapter":
            selected_filter = create_dropdown_menu(selected_chapter, filter_functions[selected_chapter])
            if selected_filter != f"Select Processing {selected_chapter}":

                imgout = apply_filter(img_array, selected_chapter, selected_filter)
                st.image(imgout, caption=f"{selected_filter} Image", use_column_width=True)
