import streamlit as st
import cv2
import torch
import sys
import time
import tempfile
sys.path.insert(0,'./pages/data/11_Car_Plate_Recognition/function')
from helper import *
from utils_rotate import *
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Nhận dạng biển số xe hơi", page_icon="🚗")

st.header("Nhận dạng biển số xe hơi")

prev_frame_time = 0
new_frame_time = 0

try:
    if st.session_state["is_load_bx"] == True:
        print('Đã load model')
except:
    st.session_state["is_load_bx"] = True
    with st.spinner('Loading model..'):
        st.session_state["yolo_LP_detect"] = torch.hub.load('./pages/data/11_Car_Plate_Recognition/yolov5', 'custom', path='./pages/data/11_Car_Plate_Recognition/model/LP_detector_nano_61.pt', force_reload=True, source='local')
        st.session_state["yolo_license_plate"] = torch.hub.load('./pages/data/11_Car_Plate_Recognition/yolov5', 'custom', path='./pages/data/11_Car_Plate_Recognition/model/LP_ocr_nano_62.pt', force_reload=True, source='local')
        st.session_state["yolo_license_plate"].conf = 0.6

video_file = st.file_uploader('Upload a video', type = ['mp4', 'mov', 'mkv'])

if video_file :
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vid = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vid.isOpened():
        ret, frame = vid.read()
        plates = st.session_state["yolo_LP_detect"](frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        for plate in list_plates:
            flag = 0
            x = int(plate[0])
            y = int(plate[1])
            w = int(plate[2] - plate[0])
            h = int(plate[3] - plate[1]) 
            crop_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = read_plate(st.session_state["yolo_license_plate"], deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(color_coverted, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        stframe.image(color_coverted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
