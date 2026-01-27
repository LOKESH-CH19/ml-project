"""
Configuration file for Sign Language Detection Project
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Dataset configuration
DATASET_URLS = {
    'asl_alphabet': 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet',
    'asl_numbers': 'https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset',
}

# Dataset paths (actual locations after extraction)
ALPHABET_TRAIN_PATH = os.path.join(RAW_DATA_DIR, 'asl_alphabet', 'asl_alphabet_train', 'asl_alphabet_train')
ALPHABET_TEST_PATH = os.path.join(RAW_DATA_DIR, 'asl_alphabet', 'asl_alphabet_test', 'asl_alphabet_test')
NUMBERS_PATH = os.path.join(RAW_DATA_DIR, 'asl_numbers', 'American Sign Language Digits Dataset')
CONTROL_GESTURES_PATH = os.path.join(RAW_DATA_DIR, 'control_gestures')

# Gesture classes
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # All 26 letters (Y dual-use for mode switch)
NUMBERS = list('0123456789')

# Map existing dataset gestures to control functions
# The ASL alphabet dataset includes: 'del', 'space', 'nothing' + we use 'Y' for mode
CONTROL_MAPPINGS = {
    'space': 'SPACE',         # Space between words (from dataset)
    'del': 'BACKSPACE',       # Delete/backspace (from dataset)  
    'nothing': 'SEND',        # Speak message (no hand visible)
    'Y': 'MODE_SWITCH',       # Y gesture (hang loose) switches ABC ↔ 123
}

# We'll use these existing gestures as controls
CONTROLS = ['space', 'del', 'nothing']  # Y is in LETTERS but acts as control when detected
CONTROL_ACTIONS = list(CONTROL_MAPPINGS.values())  # ['SPACE', 'BACKSPACE', 'SEND', 'MODE_SWITCH']

ALL_CLASSES = LETTERS + NUMBERS + CONTROLS
NUM_CLASSES = len(ALL_CLASSES)

# Model configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Detection configuration
CONFIDENCE_THRESHOLD = 0.85
HOLD_TIME = 1.5  # seconds - gesture must be held for this duration
COOLDOWN_TIME = 0.5  # seconds - pause after detection
FPS = 30

# MediaPipe configuration
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Mode configuration
MODES = {
    'LETTER': {'classes': LETTERS, 'display': 'ABC'},
    'NUMBER': {'classes': NUMBERS, 'display': '123'},
}

# Send gesture - use the Speak button instead
# Mode switch gesture - "nothing" (no hand visible)

# Quick Phrases
DEFAULT_QUICK_PHRASES = [
    "Hello, how are you?",
    "Thank you very much",
    "I need help",
    "Yes",
    "No",
    "Please wait",
    "Nice to meet you",
    "Good morning",
    "Good night",
    "See you later",
]

# Text-to-Speech configuration
TTS_RATE = 150  # words per minute
TTS_VOLUME = 0.9

# Contact system
MAX_CONTACTS = 20
CONTACTS_FILE = os.path.join(BASE_DIR, 'contacts.json')

# History and logs
MAX_HISTORY = 50
PHRASES_FILE = os.path.join(BASE_DIR, 'quick_phrases.json')
HISTORY_FILE = os.path.join(BASE_DIR, 'conversation_history.json')

# GUI configuration
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
