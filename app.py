import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import tensorflow as tf
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Load the model
model = tf.keras.models.load_model('sign_language_landmark_best.h5')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Predict
            prediction = model.predict(np.array([landmarks]), verbose=0)
            predicted_class = np.argmax(prediction)
            
            # Assume classes 0-25 correspond to A-Z
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            predicted_letter = letters[predicted_class]
            
            cv2.putText(img, f"Sign: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Real-Time Hand Sign Detector")
webrtc_streamer(key="hand-sign", video_frame_callback=video_frame_callback)