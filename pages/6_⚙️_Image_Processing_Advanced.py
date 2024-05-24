import cv2
import streamlit as st
import numpy as np

def invert_image(image):
    inverted_image = 255 - image
    return inverted_image

def apply_logarithm(image, c):
    log_image = c * np.log(1 + image)
    log_image = np.uint8(log_image)
    return log_image

def apply_power(image, gamma):
    power_image = np.power(image, gamma)
    power_image = np.uint8(power_image)
    return power_image

def apply_piecewise_linear(image, x_min, x_max, y_min, y_max):
    alpha = (y_max - y_min) / (x_max - x_min)
    beta = y_min - alpha * x_min
    linear_image = np.clip(alpha * image + beta, 0, 255)
    linear_image = np.uint8(linear_image)
    return linear_image

# Định nghĩa các chức năng xử lí ảnh
IMAGE_FUNCTIONS = {
    'Âm ảnh (Negative)': invert_image,
    'Logarithm': apply_logarithm,
    'Lũy thừa': apply_power,
    'Biến đổi tuyến tính từng phần': apply_piecewise_linear,
}

st.set_page_config(page_title="Xử lý ảnh tăng cường", page_icon="⚙️")

st.header("Xử lý ảnh tăng cường")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "bmp"])

if uploaded_file is not None:
    # Đọc ảnh từ file
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Hiển thị ảnh gốc
    st.subheader('Ảnh gốc')
    st.image(image, channels='RGB')
    
    # Chọn chức năng xử lí ảnh
    selected_function = st.selectbox('Chọn chức năng', list(IMAGE_FUNCTIONS.keys()))
    
    if selected_function != 'Âm ảnh (Negative)':
        # Hiển thị các thông số cần thiết cho chức năng được chọn
        if selected_function == 'Logarithm':
            c = st.slider('Hệ số c', 0.1, 10.0, 1.0)
        elif selected_function == 'Lũy thừa':
            gamma = st.slider('Số mũ gamma', 0.1, 10.0, 1.0)
        elif selected_function == 'Biến đổi tuyến tính từng phần':
            x_min = st.slider('x_min', 0, 255, 0)
            x_max = st.slider('x_max', 0, 255, 255)
            y_min = st.slider('y_min', 0, 255, 0)
            y_max = st.slider('y_max', 0, 255, 255)
        elif selected_function == 'Local Histogram':
            block_size = st.slider('Kích thước block', 3, 21, 7, step=2)
            constant = st.slider('Giá trị hằng số', 0.0, 10.0, 1.0)
        
        # Áp dụng chức năng xử lí ảnh
        if selected_function == 'Logarithm':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, c)
        elif selected_function == 'Lũy thừa':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, gamma)
        elif selected_function == 'Biến đổi tuyến tính từng phần':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, x_min, x_max, y_min, y_max)
        elif selected_function == 'Local Histogram':
            processed_image = IMAGE_FUNCTIONS[selected_function](image, block_size, constant)
        else:
            processed_image = IMAGE_FUNCTIONS[selected_function](image)
        
        # Hiển thị ảnh đã xử lí
        st.subheader('Ảnh đã xử lí')
        st.image(processed_image, channels='RGB')
    else:
        # Áp dụng chức năng âm ảnh
        inverted_image = IMAGE_FUNCTIONS[selected_function](image)
        # Hiển thị ảnh âm ảnh
        st.subheader('Ảnh âm ảnh')
        st.image(inverted_image, channels='RGB')
