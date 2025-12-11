"""
Data Preprocessing Module
Handles data loading, augmentation, and preparation for training
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import config

class DataPreprocessor:
    """Preprocess sign language data"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.label_encoder = LabelEncoder()
    
    def extract_hand_landmarks(self, image):
        """
        Extract hand landmarks from image using MediaPipe
        Returns normalized landmark coordinates
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get landmarks for first hand (or combine both hands)
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks[:1]:  # First hand only
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def augment_image(self, image):
        """Apply data augmentation"""
        augmented_images = [image]
        
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))
        
        # Brightness adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        
        # Increase brightness
        hsv[:,:,2] = hsv[:,:,2] * 1.2
        hsv[:,:,2][hsv[:,:,2] > 255] = 255
        bright = np.array(hsv, dtype=np.uint8)
        augmented_images.append(cv2.cvtColor(bright, cv2.COLOR_HSV2BGR))
        
        # Decrease brightness
        hsv[:,:,2] = hsv[:,:,2] * 0.8
        dark = np.array(hsv, dtype=np.uint8)
        augmented_images.append(cv2.cvtColor(dark, cv2.COLOR_HSV2BGR))
        
        # Rotation
        h, w = image.shape[:2]
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented_images.append(rotated)
        
        return augmented_images
    
    def load_dataset_from_directory(self, data_dir, augment=True, use_landmarks=False, max_samples_per_class=500):
        """
        Load dataset from directory structure (memory-efficient version)
        
        Args:
            data_dir: Root directory containing class subdirectories
            augment: Whether to apply data augmentation
            use_landmarks: Whether to extract MediaPipe landmarks (True) or use raw images (False)
            max_samples_per_class: Maximum samples to load per class (to prevent memory issues)
        """
        images = []
        landmarks_list = []
        labels = []
        
        class_names = sorted([d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))])
        
        print(f"\n📂 Loading data from: {data_dir}")
        print(f"Found {len(class_names)} classes: {class_names}")
        print(f"⚠️  Loading max {max_samples_per_class} samples per class to save memory")
        
        for class_name in tqdm(class_names, desc="Loading classes"):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples per class to prevent memory issues
            if len(image_files) > max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Resize image
                image = cv2.resize(image, config.IMAGE_SIZE)
                
                # No augmentation to save memory (we have enough samples)
                if use_landmarks:
                    # Extract landmarks
                    landmarks = self.extract_hand_landmarks(image)
                    if landmarks is not None:
                        landmarks_list.append(landmarks)
                        labels.append(class_name)
                else:
                    # Use raw image - convert to float32 immediately to save memory
                    images.append(image.astype(np.float32) / 255.0)
                    labels.append(class_name)
        
        if use_landmarks:
            X = np.array(landmarks_list, dtype=np.float32)
        else:
            X = np.array(images, dtype=np.float32)  # Use float32 instead of float64
        
        y = np.array(labels)
        
        print(f"✅ Loaded {len(y)} samples")
        print(f"Data shape: {X.shape}")
        print(f"Memory usage: ~{X.nbytes / (1024**2):.1f} MB")
        
        return X, y, class_names
    
    def load_landmarks_from_csv(self, data_dir, max_samples_per_class=200):
        """
        Load hand landmarks from CSV files (for ASL numbers dataset)
        
        Args:
            data_dir: Root directory containing class subdirectories with CSV files
            max_samples_per_class: Maximum samples per class
        
        Returns:
            X: Landmark coordinates (63 features per sample)
            y: Labels
            class_names: List of class names
        """
        landmarks_list = []
        labels = []
        
        class_names = sorted([d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))])
        
        print(f"\n📂 Loading landmarks from CSV: {data_dir}")
        print(f"Found {len(class_names)} classes: {class_names}")
        print(f"⚠️  Loading max {max_samples_per_class} samples per class")
        
        for class_name in tqdm(class_names, desc="Loading CSV classes"):
            class_dir = os.path.join(data_dir, class_name)
            csv_files = [f for f in os.listdir(class_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                csv_path = os.path.join(class_dir, csv_file)
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_path)
                    
                    # Extract landmark columns (x00-z20 = 63 columns)
                    landmark_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z')) and col[1:].isdigit()]
                    
                    if len(landmark_cols) < 63:  # Need 21 landmarks × 3 coords
                        continue
                    
                    # Limit samples
                    samples = df[landmark_cols].values[:max_samples_per_class]
                    
                    for sample in samples:
                        landmarks_list.append(sample.astype(np.float32))
                        labels.append(class_name)
                        
                        if len([l for l in labels if l == class_name]) >= max_samples_per_class:
                            break
                    
                except Exception as e:
                    print(f"   ⚠️  Error reading {csv_file}: {e}")
                    continue
        
        X = np.array(landmarks_list, dtype=np.float32)
        y = np.array(labels)
        
        print(f"✅ Loaded {len(y)} samples from CSV")
        print(f"Data shape: {X.shape}")
        print(f"Memory usage: ~{X.nbytes / (1024**2):.1f} MB")
        
        return X, y, class_names
    
    def prepare_combined_dataset(self, augment_controls=True):
        """
        Combine all datasets (letters, numbers, controls)
        Note: Uses landmarks for all data to ensure consistent dimensions
        """
        print("\n" + "="*70)
        print("PREPARING COMBINED DATASET")
        print("="*70)
        
        all_X = []
        all_y = []
        all_classes = []
        
        # Check which datasets are available
        asl_alphabet_dir = config.ALPHABET_TRAIN_PATH
        asl_numbers_dir = config.NUMBERS_PATH
        control_gestures_dir = config.CONTROL_GESTURES_PATH
        
        # Load ASL Alphabet with landmark extraction (if exists)
        if os.path.exists(asl_alphabet_dir):
            print("\n1️⃣  Loading ASL Alphabet data (extracting landmarks)...")
            X_letters, y_letters, classes_letters = self.load_dataset_from_directory(
                asl_alphabet_dir, augment=False, use_landmarks=True, max_samples_per_class=1000
            )
            
            # Include A-Z letters AND control gestures (space, del, nothing)
            valid_classes = config.LETTERS + config.CONTROLS
            mask = np.isin(y_letters, valid_classes)
            X_letters = X_letters[mask]
            y_letters = y_letters[mask]
            
            if len(X_letters) > 0:
                all_X.append(X_letters)
                all_y.append(y_letters)
                all_classes.extend([c for c in classes_letters if c in valid_classes])
                print(f"   ✅ Loaded {len(X_letters)} samples (26 letters + {len(config.CONTROLS)} controls)")
            else:
                print(f"   ⚠️  No valid samples found")
        
        # Load ASL Numbers from CSV (if exists)
        if os.path.exists(asl_numbers_dir):
            print("\n2️⃣  Loading ASL Numbers data (from CSV landmarks)...")
            X_numbers, y_numbers, classes_numbers = self.load_landmarks_from_csv(
                asl_numbers_dir, max_samples_per_class=500
            )
            
            if len(X_numbers) > 0:
                all_X.append(X_numbers)
                all_y.append(y_numbers)
                all_classes.extend(classes_numbers)
                print(f"   ✅ Loaded {len(X_numbers)} samples")
        
        # Skip separate control gesture collection - we're using existing dataset gestures
        print("\n3️⃣  Using built-in control gestures from alphabet dataset")
        
        # Combine all data
        if len(all_X) == 0:
            print("\n❌ No datasets found!")
            print("\nPlease download datasets first:")
            print("  python utils/dataset_downloader.py")
            print("  python collect_data.py")
            return None
        
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_combined)
        
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Total samples: {len(X_combined)}")
        print(f"Total classes: {len(np.unique(y_combined))}")
        print(f"Image shape: {X_combined[0].shape}")
        print(f"\nClass distribution:")
        unique, counts = np.unique(y_combined, return_counts=True)
        for cls, count in sorted(zip(unique, counts)):
            print(f"  {cls}: {count} samples")
        
        return X_combined, y_encoded, y_combined, self.label_encoder.classes_
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, class_names):
        """Save processed data"""
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        
        print("\n💾 Saving processed data...")
        
        np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
        np.save(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
        np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
        np.save(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
        
        # Save label encoder and class names
        with open(os.path.join(config.PROCESSED_DATA_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(os.path.join(config.PROCESSED_DATA_DIR, 'class_names.pkl'), 'wb') as f:
            pickle.dump(class_names, f)
        
        print("✅ Data saved successfully!")
        print(f"   Location: {config.PROCESSED_DATA_DIR}")

def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    
    # Prepare dataset
    result = preprocessor.prepare_combined_dataset(augment_controls=True)
    
    if result is None:
        return
    
    X, y_encoded, y_labels, class_names = result
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=config.VALIDATION_SPLIT, 
        random_state=42, stratify=y_encoded
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, class_names)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nNext step: Run 'python train_model.py' to train the model")

if __name__ == "__main__":
    main()
