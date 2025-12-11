# 🤟 Sign Language Detection System

A comprehensive real-time American Sign Language (ASL) detection system using deep learning and computer vision. Enables deaf/mute individuals to communicate by converting hand gestures into text and speech with **99.43% accuracy**.

## ✨ Key Features

### Core Detection
- **Real-time ASL recognition** using webcam with 99.43%+ accuracy
- **39 gesture classes**: 26 letters (A-Z) + 10 numbers (0-9) + 3 control gestures
- **MediaPipe hand tracking** with 21 landmark detection
- **Smart hold detection** (1.5s hold requirement to prevent false triggers)
- **Adjustable hold time** (0.5-3.0 seconds customizable)

### Smart Typing Features
- **Auto-capitalization** (first letter of sentences)
- **Word prediction** with 3 smart suggestions
- **Quick phrases** (0-9 hotkeys for common sentences)
- **Undo button** for instant corrections
- **Contact management** (save up to 20 contacts)
- **Conversation history** (last 50 messages)

### User Interface
- **Live camera feed** with hand landmark visualization
- **Large gesture preview** showing detected sign
- **Real-time confidence meter** with color coding
- **Hold progress bar** showing gesture confirmation
- **Mode switching** between letter (ABC) and number (123) modes
- **Text-to-speech** integration for speaking messages

### Technical Specifications
- **Model**: Dense Neural Network (landmark-based)
- **Size**: 1.23 MB (321,831 parameters)
- **Speed**: ~30 FPS real-time processing
- **Accuracy**: 99.43% on 14,830+ test samples
- **Memory**: ~1.5 GB total usage

## 📊 Supported Gestures

### Letters (A-Z) - 26 Classes
The system recognizes all American Sign Language alphabet signs:

| Letter | Gesture Description | Detection |
|--------|-------------------|-----------|
| **A** | Closed fist with thumb on side | ✅ Static |
| **B** | Flat hand, fingers together up, thumb across palm | ✅ Static |
| **C** | Curved hand forming a "C" shape | ✅ Static |
| **D** | Index finger up, other fingers touch thumb | ✅ Static |
| **E** | All fingertips touch thumb, curved inward | ✅ Static |
| **F** | Index and thumb form circle, three fingers up | ✅ Static |
| **G** | Fist sideways, index and thumb pointing horizontally | ✅ Static |
| **H** | Index and middle finger extended horizontally | ✅ Static |
| **I** | Pinky finger up, other fingers closed | ✅ Static |
| **J** | Pinky up (motion: draw "J" in air) | ✅ Static |
| **K** | Index and middle in V-shape, thumb between them | ✅ Static |
| **L** | Index finger up, thumb out ("L" shape) | ✅ Static |
| **M** | Three fingers over thumb, pinky down | ✅ Static |
| **N** | Two fingers over thumb, others down | ✅ Static |
| **O** | All fingertips touch to form circle | ✅ Static |
| **P** | Like K but pointing downward | ✅ Static |
| **Q** | Like G but pointing downward | ✅ Static |
| **R** | Index and middle finger crossed | ✅ Static |
| **S** | Closed fist with thumb across fingers | ✅ Static |
| **T** | Thumb between index and middle finger | ✅ Static |
| **U** | Index and middle finger together up | ✅ Static |
| **V** | Index and middle apart in V (peace sign) | ✅ Static |
| **W** | Index, middle, and ring finger up | ✅ Static |
| **X** | Index finger bent like a hook | ✅ Static |
| **Y** | Pinky and thumb out ("hang loose") | ✅ Static |
| **Z** | Index draws "Z" in air | ✅ Static |

**Note**: ASL letters are case-insensitive. The system auto-capitalizes the first letter of sentences.

### Numbers (0-9) - 10 Classes

| Number | Gesture Description | Visual |
|--------|-------------------|--------|
| **0** | O shape with thumb and fingers forming circle | ⭕ |
| **1** | Index finger pointing up | ☝️ |
| **2** | Index and middle finger up (V-shape) | ✌️ |
| **3** | Thumb, index, and middle finger up | 🤟 |
| **4** | Four fingers up (no thumb) | 🖐️ |
| **5** | All five fingers spread open | ✋ |
| **6** | Pinky and thumb touch, three fingers up | 🤙 |
| **7** | Ring and thumb touch, other fingers up | 👌 |
| **8** | Middle and thumb touch, other fingers up | 🤌 |
| **9** | Index and thumb touch, other fingers up | 👌 |

### Control Gestures - 3 Classes

| Gesture | Action | Description | Usage |
|---------|--------|-------------|-------|
| **space** | Add Space | Open palm facing camera (like waving) | 🤚 Separates words |
| **del** | Backspace | Special deletion gesture from dataset | ⬅️ Remove last character |
| **nothing** | Send/Speak | No hand visible in frame | 🔊 Speak message via TTS |

### Mode Switching

**Switch between Letter and Number modes:**
- Show **"0"** gesture twice quickly (within 2 seconds)
- **Letter Mode (ABC)**: Blue indicator - types A-Z letters
- **Number Mode (123)**: Orange indicator - types 0-9 numbers

## 🎯 How to Use

### Starting the Application
```bash
# Launch the application
python app.py
```

### Step-by-Step Usage

#### 1. **Position Your Hand**
- Sit in front of the webcam
- Keep hand clearly visible in the camera frame
- Adequate lighting is important
- Distance: 1-2 feet from camera

#### 2. **Make a Gesture**
- Form the ASL sign for a letter/number
- Keep the gesture steady and clear
- Watch the large gesture preview on screen

#### 3. **Hold for Confirmation**
- Hold the gesture steady for 1.5 seconds (default)
- Watch the hold progress bar fill up
- Green confidence bar indicates good detection

#### 4. **Character Appears**
- Character is added to the text display
- Continue with next gestures to build words

#### 5. **Using Control Gestures**

**Add a Space:**
1. Show "space" gesture (open palm)
2. Hold for 1.5 seconds
3. Space appears between words

**Delete a Character:**
1. Show "del" gesture
2. Hold for 1.5 seconds
3. Last character is removed

**Speak the Message:**
1. Remove your hand from view ("nothing" gesture)
2. Hold hand away for 1.5 seconds
3. Text-to-speech reads your message aloud

#### 6. **Quick Features**

**Word Predictions:**
- Suggestions appear automatically as you type
- Click a suggestion to complete the word

**Undo Button:**
- Click orange "↶ Undo" button
- Instantly removes last character

**Adjust Hold Time:**
- Use the spinbox to change hold duration
- Range: 0.5 to 3.0 seconds
- Shorter = faster typing (more errors)
- Longer = more accurate (slower typing)

**Quick Phrases:**
- Double-click phrases in the list
- Instantly inserts common sentences

## 📋 Requirements

```txt
Python 3.12+
tensorflow==2.18.0 (or tensorflow-cpu==2.18.0)
opencv-python>=4.8.1.78
mediapipe>=0.10.9
numpy>=1.24.3
pandas>=2.1.4
matplotlib>=3.8.2
scikit-learn>=1.3.2
pyttsx3>=2.90
Pillow>=10.1.0
tqdm>=4.66.1
kaggle>=1.8.2
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Hand_Sign_Language
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**For Windows with TensorFlow issues:**
```bash
pip install tensorflow-cpu==2.18.0
```

### 3. Setup Kaggle API (First Time Only)
```bash
# Get your API token from https://www.kaggle.com/settings
# Save kaggle.json to: C:\Users\<your-username>\.kaggle\kaggle.json (Windows)
# Or: ~/.kaggle/kaggle.json (Linux/Mac)
```

### 4. Download & Preprocess Data
```bash
# Preprocessing extracts landmarks from images
python preprocess_data.py
# Takes ~15 minutes, creates ~34,000 training samples
```

### 5. Train the Model
```bash
python train_model.py
# Takes ~20 minutes, achieves 99.43%+ accuracy
```

### 6. Run the Application
```bash
python app.py
# Select the trained model when prompted
```

## 📁 Project Structure

```
Hand_Sign_Language/
│
├── app.py                      # Main GUI application
├── config.py                   # Configuration settings
├── model.py                    # Neural network architectures
├── train_model.py             # Model training pipeline
├── preprocess_data.py         # Data preprocessing
├── smart_typing.py            # Smart typing engine
├── features.py                # Quick phrases, contacts, history
├── collect_data.py            # Manual gesture data collection
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # Downloaded datasets
│   │   ├── asl_alphabet/      # 87,000 letter images
│   │   ├── asl_numbers/       # 2,000 number landmarks (CSV)
│   │   └── control_gestures/  # Optional manual collection
│   └── processed/             # Preprocessed landmark data
│       ├── X_train.npy        # Training features
│       ├── X_test.npy         # Test features
│       ├── y_train.npy        # Training labels
│       ├── y_test.npy         # Test labels
│       ├── class_names.pkl    # Class name mapping
│       └── label_encoder.pkl  # Label encoder
│
├── models/
│   └── sign_language_landmark_best.h5  # Trained model (1.23 MB)
│
├── logs/                      # Training logs and visualizations
│   ├── training_history.png  # Accuracy/loss plots
│   └── tensorboard/           # TensorBoard logs
│
├── utils/
│   └── dataset_downloader.py # Dataset download helper
│
└── docs/
    ├── README.md              # This file
    ├── QUICKSTART.md          # Quick start guide
    ├── PROJECT_OVERVIEW.md    # Technical overview
    ├── IMPROVEMENTS.md        # Recent improvements
    ├── START_HERE.md          # Setup instructions
    ├── TENSORFLOW_FIX.md      # Windows TensorFlow fixes
    └── NEXT_STEPS.md          # Post-setup guide
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model settings
IMAGE_SIZE = (128, 128)         # Landmark input is 63 features
BATCH_SIZE = 32                 # Training batch size
EPOCHS = 50                     # Maximum training epochs
LEARNING_RATE = 0.001           # Adam optimizer learning rate

# Detection settings
CONFIDENCE_THRESHOLD = 0.85     # Minimum confidence for detection
HOLD_TIME = 1.5                 # Seconds to hold gesture
COOLDOWN_TIME = 0.5             # Seconds between detections

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.7  # Hand detection confidence
MIN_TRACKING_CONFIDENCE = 0.5   # Hand tracking confidence
```

## 🎓 Training Details

### Dataset Composition
- **ASL Alphabet**: 29,000 samples (1000 per class × 29 classes)
  - Source: Kaggle "grassknoted/asl-alphabet"
  - Includes: A-Z letters + space, del, nothing
- **ASL Numbers**: 5,000 samples (500 per class × 10 classes)
  - Source: Kaggle "rayeed045/american-sign-language-digit-dataset"
  - Pre-extracted MediaPipe landmarks (CSV format)

### Model Performance
```
Model: "sequential"
Total params: 321,831 (1.23 MB)
Trainable params: 319,527 (1.22 MB)
Non-trainable params: 2,304 (9.00 KB)

Test Accuracy: 99.43%
Test Loss: 0.0234
Training Time: ~20 minutes (CPU)
Inference Speed: ~30 FPS
```

## 🐛 Troubleshooting

### Camera Not Working
- Grant camera permissions in Windows Settings
- Close other apps using the camera
- Run `python verify_setup.py` to test

### Low Detection Accuracy
- Ensure good lighting
- Keep hand centered in frame
- Hold gestures steady for full 1.5 seconds
- Adjust hold time if needed

### Model Not Found
```bash
python train_model.py
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Dynamic gesture detection (motion-based J, Z)
- Two-hand gesture support
- Multi-language sign language (BSL, ISL, etc.)
- Mobile app deployment

## 📄 License

Educational project. Dataset credits:
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [ASL Numbers Dataset](https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset)

---

**Built with ❤️ to help the deaf and mute community communicate effectively**

🎯 **Current Status**: Production-ready with 99.43% accuracy
# SignLanguageDetection
