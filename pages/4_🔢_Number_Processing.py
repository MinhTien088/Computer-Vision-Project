import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore
import numpy as np
import random
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

def generate_random_picture():
    image = np.zeros((10 * 28, 10 * 28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(0, 100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x * 28 : (x + 1) * 28, y * 28 : (y + 1) * 28] = sample[:, :, 0]
    return image, data


if "is_load" not in st.session_state:
    # load model

    model_architecture = os.path.join(curr_dir, "data/4_Number_Processing/digit_config.json")
    model_weights = os.path.join(curr_dir, "data/4_Number_Processing/digit_weight.h5")
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    OPTIMIZER = tf.keras.optimizers.Adam()
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    st.session_state.model = model

    # load data
    (_, _), (X_test, y_test) = datasets.mnist.load_data()
    X_test = X_test.reshape((10000, 28, 28, 1))
    st.session_state.X_test = X_test

    st.session_state.is_load = True
    print("Load model và data lần đầu")
else:
    print("Đã load model và data")

if st.button("Tạo ảnh và Nhận dạng"):
    image, data = generate_random_picture()
    data = data / 255.0
    data = data.astype("float32")
    result = st.session_state.model.predict(data)
    dem = 0
    s = ""
    for x in result:
        s = s + "%d " % (np.argmax(x))
        dem = dem + 1
        if (dem % 10 == 0) and (dem < 100):
            s = s + "\n"
    st.image(image)
    st.text(s)
