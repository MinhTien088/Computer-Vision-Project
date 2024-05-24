import cv2 as cv
import streamlit as st
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Nhận dạng chuyển động", page_icon="🚨")

st.header("Nhận dạng chuyển động")

FRAME_WINDOW = st.image([])
motion_text = st.empty()

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
    motion_text.text("")

if st.button("Rerun"):
    st.session_state.stop = False

for i in range(10):
    ret, frame = cap.read()
    
if st.session_state.stop == False:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    last_frame = gray

# Placeholder for displaying the video frames
while True:
    if st.session_state.stop:
        break

    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (25, 25), 0)
    abs_img = cv.absdiff(last_frame, gray)
    last_frame = gray

    _, img_mask = cv.threshold(abs_img, 30, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        area = cv.contourArea(contour)
        if cv.contourArea(contour) < 900:
            continue

        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.putText(frame, str(area), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        motion_detected = True

    # Convert the frame to RGB and display it using Streamlit
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    if motion_detected:
        motion_text.text("Có chuyển động")
    else:
        motion_text.text("Không có chuyển động")
