import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array  # type: ignore
# from keras.preprocessing.image import img_to_array
import cv2 as cv
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Nhận dạng cảm xúc", page_icon="🥹")

st.header("Nhận dạng cảm xúc")

FRAME_WINDOW = st.image([])

cap = cv.VideoCapture(0)

if "stop" not in st.session_state:
    st.session_state.stop = False
    stop = False

if st.button("Stop"):
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

print("Trạng thái Stop:", st.session_state.stop)

if "frame_stop" not in st.session_state:
    frame_stop = cv.imread(os.path.join(curr_dir, "data/1_Face_Recognition/stop.png"))
    st.session_state.frame_stop = frame_stop
    print("Đã stop")

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels="BGR")

if st.button("Rerun"):
    st.session_state.stop = False

try:
    if st.session_state["LoadModel6"] == True:
        print("Đã load model")
        pass
except:
    class_path = os.path.join(curr_dir, "data/6_Emotion_Recognition/haarcascade_frontalface_default.xml")
    st.session_state["CascadeClassifier"] = cv.CascadeClassifier(class_path)
    model_path = os.path.join(curr_dir, "data/6_Emotion_Recognition/model.h5")
    st.session_state["Model"] = load_model(model_path)
    st.session_state["LoadModel6"] = True
    print("Load model lần đầu")

face_classifier = st.session_state["CascadeClassifier"]

model = st.session_state["Model"]

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv.putText(
                frame,
                label,
                label_position,
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    FRAME_WINDOW.image(frame, channels="BGR")
    cv.destroyAllWindows()

cap.release()
