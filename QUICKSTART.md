# 🚀 Quick Start Guide

## Complete Setup in 5 Steps

### Step 1: Install Dependencies (5 minutes)
```powershell
# Navigate to project directory
cd d:\Hand_Sign_Language

# Install required packages
pip install -r requirements.txt

# Install Kaggle CLI for datasets
pip install kaggle
```

### Step 2: Setup Kaggle API (2 minutes)
1. Go to https://www.kaggle.com/settings
2. Click **"Create New API Token"**
3. Save `kaggle.json` to: `C:\Users\<YourUsername>\.kaggle\`
4. Create the folder if it doesn't exist

### Step 3: Download Datasets (10-20 minutes)

```powershell
# View dataset information
python utils/dataset_downloader.py

# Download ASL Alphabet dataset (1.1 GB)
kaggle datasets download -d grassknoted/asl-alphabet -p data/raw

# Download ASL Numbers dataset (50 MB)
kaggle datasets download -d rayeed045/american-sign-language-digit-dataset -p data/raw

# Extract the zip files to data/raw/
```

**Alternative: Manual Download**
1. Visit dataset URLs (shown in dataset_downloader.py output)
2. Download and extract to `data/raw/asl_alphabet/` and `data/raw/asl_numbers/`

### Step 4: Collect Control Gesture Data (10 minutes)

```powershell
# Run data collection tool
python collect_data.py

# Follow on-screen instructions:
# - Choose option 1 (Collect all control gestures)
# - Use default 500 samples per gesture
# - Hold each gesture steady when collecting
```

**Gestures to collect:**
1. SPACE (open palm)
2. SEND (thumbs up)
3. BACKSPACE (thumbs down)
4. CLEAR (closed fist)
5. MODE_SWITCH (pinky extended)
6. PAUSE (peace sign)

### Step 5: Preprocess, Train & Run (30-60 minutes)

```powershell
# Preprocess all datasets
python preprocess_data.py

# Train the model (30-60 minutes depending on hardware)
python train_model.py
# Choose model type: 1 (Custom CNN)
# Use defaults for epochs and batch size

# Run the application
python app.py
```

---

## 🎯 Using the Application

### Basic Workflow

1. **Launch app**: `python app.py`
2. **Position hand** in front of webcam
3. **Show gesture** and hold for 1.5 seconds
4. **See character** appear in text box
5. **Continue** building your message
6. **Show thumbs up** to speak the message

### Common Use Cases

**Type "HELLO":**
```
H gesture (hold 1.5s) → H
E gesture (hold 1.5s) → E
L gesture (hold 1.5s) → L
L gesture (hold 1.5s) → L
O gesture (hold 1.5s) → O
Result: "Hello" (auto-capitalized)
```

**Type "HI 5":**
```
H gesture → H
I gesture → I
Open palm → (space)
Pinky out → (mode switches to NUMBER)
5 gesture → 5
Result: "Hi 5"
```

**Use Quick Phrase:**
```
Double-click phrase in list
Or: Show number gesture + thumbs up
```

---

## 📊 Expected Results

### Dataset Sizes (After Collection)
- Letters (A-Z): ~87,000 images
- Numbers (0-9): ~2,000 images
- Controls: ~3,000 images (500 per gesture × 6)
- **Total: ~92,000 images**

### Training Time (Approximate)
- **CPU**: 2-3 hours
- **GPU (GTX 1060+)**: 20-40 minutes
- **GPU (RTX 2060+)**: 10-20 minutes

### Model Performance
- **Accuracy**: 90-95%
- **Top-3 Accuracy**: 97-99%
- **Real-time FPS**: 20-30

---

## 🔧 Troubleshooting

### Issue: Kaggle download fails
**Solution:**
```powershell
# Check kaggle.json location
dir C:\Users\$env:USERNAME\.kaggle\

# If missing, create folder
mkdir C:\Users\$env:USERNAME\.kaggle\

# Download API token from Kaggle and move it there
```

### Issue: Camera not detected
**Solution:**
```python
# Edit app.py line with cv2.VideoCapture
# Try different camera indices:
self.cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Issue: Out of memory during training
**Solution:**
```python
# Edit config.py
BATCH_SIZE = 16  # Reduce from 32
# Or train with smaller dataset initially
```

### Issue: Low accuracy
**Solution:**
1. Collect more control gesture data (1000+ samples)
2. Ensure good lighting during collection
3. Train for more epochs (increase to 75-100)
4. Try transfer learning model (MobileNetV2)

---

## 🎓 Tips for Best Results

### Data Collection
✅ **DO:**
- Collect in different lighting conditions
- Vary hand positions and angles
- Use different backgrounds
- Include both hands (where applicable)
- Hold gestures steady

❌ **DON'T:**
- Rush through collection
- Use only one lighting condition
- Keep exact same hand position
- Block the hand with objects

### During Detection
✅ **DO:**
- Ensure good lighting
- Keep hand fully visible
- Hold gestures for full 1.5 seconds
- Wait for green confirmation
- Position hand at consistent distance

❌ **DON'T:**
- Move hand while detecting
- Cover hand landmarks
- Rush between gestures (cooldown: 0.5s)
- Use with poor/dark lighting

---

## 📚 Next Steps

After successful setup:

1. **Test all gestures**: Go through A-Z, 0-9, and controls
2. **Customize quick phrases**: Edit the 10 default phrases
3. **Add contacts**: Store frequently contacted people
4. **Review history**: Check conversation logs
5. **Adjust settings**: Fine-tune in `config.py`

---

## 🆘 Getting Help

If you encounter issues:

1. Check **Troubleshooting** section above
2. Review **README.md** for detailed documentation
3. Verify all dependencies are installed
4. Check Python version (3.8+ required)
5. Ensure webcam permissions are granted

---

## 🎉 Success Indicators

You'll know everything is working when:

✅ Camera feed shows in application window
✅ Hand landmarks are drawn on video
✅ Confidence bar shows >85% for gestures
✅ Text appears in message box after holding gesture
✅ Mode switches between ABC and 123
✅ Text-to-speech works when clicking Speak

---

**Enjoy your Sign Language Detection System!** 🤟

*Remember: The more diverse your training data, the better the accuracy!*
