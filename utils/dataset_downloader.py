"""
Dataset Downloader and Manager
Handles downloading and organizing ASL datasets
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DatasetDownloader:
    """Download and organize sign language datasets"""
    
    def __init__(self):
        self.data_dir = config.RAW_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_file(self, url, destination):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to: {extract_to}")
    
    def setup_datasets(self):
        """
        Setup dataset directories
        
        DATASET INFORMATION:
        
        1. ASL Alphabet Dataset (Kaggle)
           URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
           - 87,000 images (200x200 pixels)
           - 29 classes: A-Z + SPACE + DELETE + NOTHING
           - Multiple hands, backgrounds, lighting
           
        2. ASL Numbers Dataset (Kaggle)
           URL: https://www.kaggle.com/datasets/rayeed045/american-sign-language-digit-dataset
           - 2,062 images
           - 10 classes: 0-9
           
        3. Custom Control Gestures (To be collected)
           - We'll create a collection tool for control gestures
        
        DOWNLOAD INSTRUCTIONS:
        1. Install Kaggle CLI: pip install kaggle
        2. Get Kaggle API credentials from https://www.kaggle.com/settings
        3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)
        4. Run the download commands below
        """
        
        print("=" * 70)
        print("DATASET SETUP INSTRUCTIONS")
        print("=" * 70)
        print("\n📦 REQUIRED DATASETS:\n")
        
        print("1. ASL ALPHABET DATASET")
        print("   Source: Kaggle - grassknoted/asl-alphabet")
        print("   Size: ~1.1 GB (87,000 images)")
        print("   Classes: A-Z + control gestures")
        print("   Download command:")
        print("   kaggle datasets download -d grassknoted/asl-alphabet -p data/raw")
        print()
        
        print("2. ASL NUMBERS DATASET")
        print("   Source: Kaggle - rayeed045/american-sign-language-digit-dataset")
        print("   Size: ~50 MB (2,062 images)")
        print("   Classes: 0-9")
        print("   Download command:")
        print("   kaggle datasets download -d rayeed045/american-sign-language-digit-dataset -p data/raw")
        print()
        
        print("3. ALTERNATIVE: Pre-processed Dataset (Recommended)")
        print("   Combined ASL Alphabet + Digits Dataset")
        print("   URL: https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset")
        print("   Download command:")
        print("   kaggle datasets download -d debashishsau/aslamerican-sign-language-aplhabet-dataset -p data/raw")
        print()
        
        print("=" * 70)
        print("\n🔧 SETUP STEPS:\n")
        print("1. Install Kaggle CLI:")
        print("   pip install kaggle")
        print()
        print("2. Get Kaggle API Token:")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to: C:\\Users\\<YourUsername>\\.kaggle\\")
        print()
        print("3. Run download commands above")
        print()
        print("4. Extract downloaded zip files to data/raw/")
        print()
        print("5. For control gestures, we'll use the dataset collection tool")
        print()
        print("=" * 70)
        
        # Create directory structure
        os.makedirs(os.path.join(self.data_dir, 'asl_alphabet'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'asl_numbers'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'control_gestures'), exist_ok=True)
        
        print("\n✅ Directory structure created!")
        print(f"   {self.data_dir}/asl_alphabet/")
        print(f"   {self.data_dir}/asl_numbers/")
        print(f"   {self.data_dir}/control_gestures/")
        
        return True

def main():
    """Run dataset setup"""
    downloader = DatasetDownloader()
    downloader.setup_datasets()
    
    print("\n" + "=" * 70)
    print("📌 NEXT STEPS:")
    print("=" * 70)
    print("1. Follow the instructions above to download datasets")
    print("2. Run: python collect_data.py (to collect control gesture data)")
    print("3. Run: python preprocess_data.py (to prepare data for training)")
    print("4. Run: python train_model.py (to train the model)")
    print("=" * 70)

if __name__ == "__main__":
    main()
