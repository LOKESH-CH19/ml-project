import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import cv2
import av
import tensorflow as tf
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Download the hand landmarker model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', model_path)

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Load the sign language model
model = tf.keras.models.load_model('sign_language_landmark_best.h5')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = hand_landmarker.detect(mp_image)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw landmarks
            for lm in hand_landmarks:
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            for connection in HAND_CONNECTIONS:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                start_x, start_y = int(start.x * img.shape[1]), int(start.y * img.shape[0])
                end_x, end_y = int(end.x * img.shape[1]), int(end.y * img.shape[0])
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks:
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