"""
Setup Verification Script
Verify that all dependencies and configurations are correct
"""

import sys
import importlib
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✅ Python version OK")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_directories():
    """Check if required directories exist"""
    dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'utils'
    ]
    
    print("\n📁 Checking directories...")
    all_exist = True
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MISSING")
            all_exist = False
    
    return all_exist

def check_camera():
    """Check if camera is accessible"""
    print("\n📷 Checking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera working (Resolution: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
                return True
            else:
                print("❌ Camera not returning frames")
                cap.release()
                return False
        else:
            print("❌ Camera not accessible")
            return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def check_kaggle():
    """Check Kaggle API setup"""
    print("\n🔑 Checking Kaggle API...")
    
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_file):
        print(f"✅ Kaggle API token found at {kaggle_file}")
        return True
    else:
        print(f"⚠️  Kaggle API token not found")
        print(f"   Expected location: {kaggle_file}")
        print(f"   Get token from: https://www.kaggle.com/settings")
        return False

def check_datasets():
    """Check if datasets are downloaded"""
    print("\n💾 Checking datasets...")
    
    datasets = {
        'ASL Alphabet': 'data/raw/asl_alphabet',
        'ASL Numbers': 'data/raw/asl_numbers',
        'Control Gestures': 'data/raw/control_gestures',
    }
    
    all_exist = True
    for name, path in datasets.items():
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            count = len(os.listdir(path))
            print(f"✅ {name}: {count} files/folders")
        else:
            print(f"❌ {name}: NOT FOUND")
            all_exist = False
    
    return all_exist

def check_processed_data():
    """Check if data is preprocessed"""
    print("\n🔧 Checking preprocessed data...")
    
    files = [
        'data/processed/X_train.npy',
        'data/processed/X_test.npy',
        'data/processed/y_train.npy',
        'data/processed/y_test.npy',
        'data/processed/class_names.pkl',
    ]
    
    all_exist = True
    for file_path in files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {os.path.basename(file_path)}: {size_mb:.1f} MB")
        else:
            print(f"❌ {os.path.basename(file_path)}: NOT FOUND")
            all_exist = False
    
    return all_exist

def check_trained_model():
    """Check if model is trained"""
    print("\n🤖 Checking trained models...")
    
    if os.path.exists('models'):
        models = [f for f in os.listdir('models') if f.endswith('.h5')]
        if models:
            for model in models:
                size_mb = os.path.getsize(os.path.join('models', model)) / (1024 * 1024)
                print(f"✅ {model}: {size_mb:.1f} MB")
            return True
        else:
            print("❌ No trained models found")
            return False
    else:
        print("❌ Models directory not found")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("🔍 SIGN LANGUAGE DETECTION - SETUP VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Check Python version
    print("\n🐍 Python Environment")
    print("-"*70)
    results['python'] = check_python_version()
    
    # Check packages
    print("\n📦 Required Packages")
    print("-"*70)
    packages = [
        ('tensorflow', 'tensorflow'),
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('pyttsx3', 'pyttsx3'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm'),
    ]
    
    package_results = []
    for pkg_name, import_name in packages:
        package_results.append(check_package(pkg_name, import_name))
    
    results['packages'] = all(package_results)
    
    # Check directories
    results['directories'] = check_directories()
    
    # Check camera
    results['camera'] = check_camera()
    
    # Check Kaggle
    results['kaggle'] = check_kaggle()
    
    # Check datasets
    results['datasets'] = check_datasets()
    
    # Check processed data
    results['processed'] = check_processed_data()
    
    # Check trained model
    results['model'] = check_trained_model()
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    status_icon = lambda x: "✅" if x else "❌"
    
    print(f"\n{status_icon(results['python'])} Python Environment")
    print(f"{status_icon(results['packages'])} Required Packages")
    print(f"{status_icon(results['directories'])} Directory Structure")
    print(f"{status_icon(results['camera'])} Camera Access")
    print(f"{status_icon(results['kaggle'])} Kaggle API")
    print(f"{status_icon(results['datasets'])} Datasets")
    print(f"{status_icon(results['processed'])} Preprocessed Data")
    print(f"{status_icon(results['model'])} Trained Model")
    
    # Next steps
    print("\n" + "="*70)
    print("📋 NEXT STEPS")
    print("="*70)
    
    if not results['packages']:
        print("\n1. Install missing packages:")
        print("   pip install -r requirements.txt")
    
    if not results['kaggle']:
        print("\n2. Setup Kaggle API:")
        print("   - Get token from https://www.kaggle.com/settings")
        print("   - Save to ~/.kaggle/kaggle.json")
    
    if not results['datasets']:
        print("\n3. Download datasets:")
        print("   python utils/dataset_downloader.py")
        print("   python collect_data.py")
    
    if not results['processed']:
        print("\n4. Preprocess data:")
        print("   python preprocess_data.py")
    
    if not results['model']:
        print("\n5. Train model:")
        print("   python train_model.py")
    
    if all([results['packages'], results['datasets'], results['processed'], results['model']]):
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nYou're ready to run the application:")
        print("   python app.py")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
