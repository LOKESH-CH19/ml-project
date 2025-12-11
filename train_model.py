"""
Model Training Script
Train the sign language detection model
"""

import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
import config
from model import create_landmark_model

class ModelTrainer:
    """Train and evaluate sign language detection model"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = None
        
        # Create directories
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data"""
        print("\n📂 Loading preprocessed data...")
        
        data_dir = config.PROCESSED_DATA_DIR
        
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        with open(os.path.join(data_dir, 'class_names.pkl'), 'rb') as f:
            self.class_names = pickle.load(f)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Number of classes: {len(self.class_names)}")
        print(f"   Input shape: {X_train.shape[1:]}")
        print(f"   Classes: {list(self.class_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_data_augmentation(self):
        """Create data augmentation layer"""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
        ])
    
    def get_callbacks(self, model_name):
        """Get training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Model checkpoint - save best model
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.MODELS_DIR, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.LOGS_DIR, f'{model_name}_{timestamp}'),
                histogram_freq=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                os.path.join(config.LOGS_DIR, f'{model_name}_{timestamp}.csv')
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   model_type='cnn', epochs=None, batch_size=None):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            model_type: Type of model to train
            epochs: Number of training epochs
            batch_size: Batch size
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        # Create model
        num_classes = len(self.class_names)
        input_shape = X_train.shape[1:]
        
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        print("="*70)
        print(f"Model type: Landmark-based DNN")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Input shape: {input_shape}")
        print("="*70)
        
        self.model = create_landmark_model(num_classes, input_shape)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n📊 Model Architecture:")
        self.model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks('sign_language_landmark')
        
        # Train model
        print("\n🚀 Starting training...")
        print("="*70)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\n📊 Evaluating model on test set...")
        print("="*70)
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
        print("="*70)
        
        return results
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📈 Training plots saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path=None):
        """Save the trained model"""
        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(config.MODELS_DIR, f'sign_language_model_{timestamp}.h5')
        
        self.model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        return model_path

def main():
    """Main training function"""
    print("\n🎯 Sign Language Detection Model Training")
    print("="*70)
    
    trainer = ModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Use landmark model (no choice needed since we're using landmarks)
    print("\n🔧 Using Landmark-based Dense Neural Network")
    print("   Optimized for MediaPipe hand landmark features")
    
    # Training configuration with defaults
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE
    
    # Train model
    history = trainer.train_model(
        X_train, y_train, X_test, y_test,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    results = trainer.evaluate_model(X_test, y_test)
    
    # Plot training history
    plot_path = os.path.join(config.LOGS_DIR, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)
    
    # Save model
    model_path = trainer.save_model()
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📦 Model saved to: {model_path}")
    print(f"📊 Logs saved to: {config.LOGS_DIR}")
    print(f"\n🎯 Final Test Accuracy: {results[1]*100:.2f}%")
    print("\nNext step: Run 'python app.py' to use the model in real-time!")
    print("="*70)

if __name__ == "__main__":
    # Import layers for data augmentation
    from tensorflow.keras import layers
    main()
