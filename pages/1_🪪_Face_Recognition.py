import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os
import tempfile

curr_dir = os.path.dirname(os.path.abspath(__file__))

def get_subdirectories(parent_directory):
    subdirectories = []
    for item in os.listdir(parent_directory):
        if os.path.isdir(os.path.join(parent_directory, item)):
            subdirectories.append(item)
    return subdirectories

st.set_page_config(page_title="Nhận dạng gương mặt", page_icon="🪪")

st.header("Nhận dạng gương mặt")

FRAME_WINDOW = st.image([])

# Initialize session state for stop if not already set
if "stop" not in st.session_state:
    st.session_state.stop = False

# Select the source of the video
source_option = st.selectbox("Source", ("Camera", "Upload Video"))

# Reset stop state when the source is changed
if "previous_source" not in st.session_state or st.session_state.previous_source != source_option:
    st.session_state.stop = False
    st.session_state.previous_source = source_option

cap = None

if source_option == "Upload Video":
    vid_file_buffer = st.file_uploader("Upload a video", type=["mp4", "mkv", "mov", "ts", "jpg"])
    if vid_file_buffer is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(vid_file_buffer.read())
            tmp_file_path = tmp_file.name
        cap = cv.VideoCapture(tmp_file_path)
else:
    cap = cv.VideoCapture(0)

if st.button("Stop"):
    st.session_state.stop = not st.session_state.stop
    if st.session_state.stop:
        if cap:
            cap.release()
    else:
        if source_option == "Upload Video" and vid_file_buffer is not None:
            cap = cv.VideoCapture(tmp_file_path)
        else:
            cap = cv.VideoCapture(0)

print("Trạng thái Stop:", st.session_state.stop)

if "frame_stop" not in st.session_state:
    frame_stop = cv.imread(os.path.join(curr_dir, "data/1_Face_Recognition/stop.png"))
    st.session_state.frame_stop = frame_stop
    print("Đã stop")

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels="BGR")

if st.button("Rerun"):
    st.session_state.stop = False
    st.session_state.previous_source = source_option

try:
    if st.session_state["LoadModel1"] == True:
        print("Đã load model và data")
        pass
except:
    parent_directory = os.path.join(curr_dir, "data/1_Face_Recognition/image")
    st.session_state["MyDict"] = get_subdirectories(parent_directory)
    model_path = os.path.join(curr_dir, "data/1_Face_Recognition/model/svc.pkl")
    st.session_state["Model"] = joblib.load(model_path)
    face_detect = os.path.join(curr_dir, "data/1_Face_Recognition/model/face_detection_yunet_2023mar.onnx")
    st.session_state["FaceDetectorYN"] = face_detect
    face_recog = os.path.join(curr_dir, "data/1_Face_Recognition/model/face_recognition_sface_2021dec.onnx")
    st.session_state["FaceRecognizerSF"] = face_recog
    st.session_state["LoadModel1"] = True
    print("Load model và data lần đầu")

svc = st.session_state["Model"]

mydict = st.session_state["MyDict"]

def visualize(input, faces, results, fps, thickness=4, fontScale=1):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(
                input,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                (0, 255, 0),
                thickness,
            )
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

            # Display recognition result on the frame with a colorful text
            result = results[idx] if idx < len(results) else "Unknown"
            cv.putText(
                input,
                result,
                (coords[0], coords[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (255, 0, 0),
                thickness,
            )

    cv.putText(
        input,
        "FPS: {:.2f}".format(fps),
        (1, 16),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

if __name__ == "__main__":
    if cap is not None and cap.isOpened():
        detector = cv.FaceDetectorYN.create(
            st.session_state["FaceDetectorYN"],
            "",
            (320, 320),
            0.9,
            0.3,
            5000,
        )

        recognizer = cv.FaceRecognizerSF.create(
            st.session_state["FaceRecognizerSF"],
            "",
        )

        tm = cv.TickMeter()

        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        results_list = []  # List to store recognition results for each frame
        while True:
            if st.session_state.stop:
                break

            hasFrame, frame = cap.read()
            if not hasFrame:
                print("No frames grabbed!")
                break

            # Inference
            tm.start()
            faces = detector.detect(frame)  # faces is a tuple
            tm.stop()

            frame_results = []  # List to store recognition results for each face in the frame
            if faces[1] is not None:
                for idx, face in enumerate(faces[1]):
                    face_align = recognizer.alignCrop(frame, face)
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    result = mydict[test_predict[0]]
                    frame_results.append(result)

            # Draw results on the input image
            visualize(frame, faces, frame_results, tm.getFPS())

            # Visualize results
            FRAME_WINDOW.image(frame, channels="BGR")

        # Display the recognition results for each face in each frame
        for i, results in enumerate(results_list):
            st.write(f"Frame {i + 1}: {results}")

        cap.release()
        cv.destroyAllWindows()
    else:
        st.warning("No video source available.")
