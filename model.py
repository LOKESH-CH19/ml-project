"""
Deep Learning Model Architecture
Neural network models for sign language classification from landmarks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import config

def create_landmark_model(num_classes, input_shape=(63,)):
    """
    Create a Dense Neural Network for landmark-based classification
    
    Args:
        num_classes: Number of output classes
        input_shape: Shape of input landmarks (63 for 21 hand landmarks × 3 coordinates)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First dense block
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second dense block
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Third dense block
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fourth dense block
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_model(num_classes, input_shape=(128, 128, 3)):
    """
    Create a custom CNN model for sign language classification
    
    Args:
        num_classes: Number of output classes
        input_shape: Shape of input images
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_transfer_learning_model(num_classes, input_shape=(128, 128, 3), 
                                   base_model='mobilenet'):
    """
    Create a transfer learning model using pre-trained networks
    
    Args:
        num_classes: Number of output classes
        input_shape: Shape of input images
        base_model: Base model to use ('mobilenet' or 'efficientnet')
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained model
    if base_model == 'mobilenet':
        base = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model == 'efficientnet':
        base = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model}")
    
    # Freeze base model layers
    base.trainable = False
    
    # Create model
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def get_model_summary(model):
    """Print model summary"""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    print("="*70)

def create_model(num_classes, model_type='cnn', input_shape=(128, 128, 3)):
    """
    Create and compile model
    
    Args:
        num_classes: Number of output classes
        model_type: Type of model ('cnn', 'mobilenet', 'efficientnet')
        input_shape: Shape of input images
    
    Returns:
        Compiled Keras model
    """
    print(f"\n🔨 Creating {model_type.upper()} model...")
    
    if model_type == 'cnn':
        model = create_cnn_model(num_classes, input_shape)
    elif model_type in ['mobilenet', 'efficientnet']:
        model = create_transfer_learning_model(num_classes, input_shape, model_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model, config.LEARNING_RATE)
    get_model_summary(model)
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")
    
    test_num_classes = 42  # 26 letters + 10 numbers + 6 controls
    
    print("1. Custom CNN Model:")
    model_cnn = create_model(test_num_classes, model_type='cnn')
    
    print("\n2. MobileNetV2 Transfer Learning Model:")
    model_mobile = create_model(test_num_classes, model_type='mobilenet')
    
    print("\n✅ All models created successfully!")
