import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

st.set_page_config(page_title="Theo dõi cử động tay", page_icon="🖐")

st.header("Theo dõi cử động tay")

# Initialize session state
if 'tracking' not in st.session_state:
    st.session_state.tracking = False

# Settings
max_num_hands = st.slider('Số tay tối đa:', 1, 4, 2, 1)
min_detection_confidence = st.slider('Độ chính xác tối thiểu:', 0.1, 1.0, 0.5, 0.1)

if st.button('Start'):
    st.session_state.tracking = True

if st.button('Stop'):
    st.session_state.tracking = False

def count_fingers(hand_landmarks):
    """
    Count the number of fingers that are up.
    """
    # Landmark indices for each finger
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcp = [2, 5, 9, 13, 17]

    fingers_up = []

    # Determine the hand is left or right
    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x:
        is_right_hand = True
    else:
        is_right_hand = False

    # Check if the thumb is up
    if is_right_hand:
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_mcp[0]].x:
            fingers_up.append(True)
        else:
            fingers_up.append(False)
    else:
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_mcp[0]].x:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

    # Check for the other fingers
    for tip, mcp in zip(finger_tips[1:], finger_mcp[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            fingers_up.append(True)
        else:
            fingers_up.append(False)

    return fingers_up

def track_hands_in_real_time(max_num_hands, min_detection_confidence):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence) as hands:
        stframe = st.empty()
        
        while cap.isOpened() and st.session_state.tracking:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Count the number of fingers up
                    fingers_up = count_fingers(hand_landmarks)
                    num_fingers_up = sum(fingers_up)
                    
                    # Display the number of fingers up on the frame
                    cv2.putText(frame, f'Fingers: {num_fingers_up}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            stframe.image(frame, channels='BGR')
        
    cap.release()

if st.session_state.tracking:
    track_hands_in_real_time(max_num_hands, min_detection_confidence)
