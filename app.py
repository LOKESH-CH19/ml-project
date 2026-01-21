import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- INSERT YOUR CLASSIFICATION LOGIC HERE ---
            # e.g., prediction = my_model.predict(landmarks)
            cv2.putText(img, "Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Hand Sign Detector")
webrtc_streamer(key="hand-sign", video_frame_callback=video_frame_callback)