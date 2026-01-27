"""
Convert TensorFlow Keras model to TensorFlow Lite format for Flutter
"""

import tensorflow as tf
import os

def convert_h5_to_tflite():
    """Convert .h5 model to .tflite format"""
    
    # Paths
    model_path = 'models/sign_language_landmark_best.h5'
    output_path = 'models/sign_language_model.tflite'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure the model file exists.")
        return
    
    print("Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nConverting to TensorFlow Lite...")
    
    # Convert with optimizations
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    h5_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\n✅ Conversion successful!")
    print(f"   Original (.h5): {h5_size:.2f} MB")
    print(f"   Converted (.tflite): {tflite_size:.2f} MB")
    print(f"   Size reduction: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
    print(f"\n📁 Output: {output_path}")
    print(f"\n🚀 Copy this file to your Flutter project:")
    print(f"   D:\\HandTalk\\handtalk\\assets\\models\\sign_language_model.tflite")

if __name__ == "__main__":
    convert_h5_to_tflite()
