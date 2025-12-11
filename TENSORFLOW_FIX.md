# ⚠️ TensorFlow Installation Fix for Windows

## Issue
TensorFlow installation fails due to Windows Long Path limitation.

## Solution Options

### Option 1: Enable Long Paths (Recommended)
1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find `LongPathsEnabled` (or create it as DWORD if missing)
4. Set value to `1`
5. Restart your computer
6. Run: `pip install tensorflow==2.18.0`

### Option 2: Enable via Group Policy (Windows 10/11 Pro)
1. Press `Win + R`, type `gpedit.msc`, press Enter
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Find "Enable Win32 long paths"
4. Set to "Enabled"
5. Apply and restart
6. Run: `pip install tensorflow==2.18.0`

### Option 3: Use PowerShell (Admin Required)
```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Then restart and run: `pip install tensorflow==2.18.0`

### Option 4: Use TensorFlow-CPU (Smaller, No GPU)
```powershell
pip install tensorflow-cpu==2.18.0
```
This version is smaller and less likely to hit path limits.

### Option 5: Use PyTorch Instead (Alternative)
If TensorFlow continues to fail, we can modify the code to use PyTorch:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
(Requires code modifications)

## Quick Test After Installation
```powershell
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"
```

## Current Status
✅ All other packages installed successfully:
- opencv-python: 4.11.0
- mediapipe: 0.10.21
- numpy, pandas, matplotlib, scikit-learn, pyttsx3, pillow, tqdm

❌ TensorFlow: Needs Windows Long Path enabled

## Next Steps After Fixing
1. Enable Long Paths (choose option above)
2. Restart computer
3. Install TensorFlow: `pip install tensorflow==2.18.0`
4. Continue with: `python utils/dataset_downloader.py`
