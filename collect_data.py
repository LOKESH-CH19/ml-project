"""
Dataset Collection Tool
Collects custom control gesture data using webcam
"""

import cv2
import mediapipe as mp
import os
import time
import numpy as np
from datetime import datetime
import config

class DataCollector:
    """Collect sign language gesture data"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.data_dir = os.path.join(config.RAW_DATA_DIR, 'control_gestures')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def create_class_directory(self, class_name):
        """Create directory for a specific class"""
        class_dir = os.path.join(self.data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir
    
    def collect_gesture_data(self, gesture_name, num_samples=500):
        """
        Collect data for a specific gesture
        
        Args:
            gesture_name: Name of the gesture (e.g., 'SPACE', 'SEND')
            num_samples: Number of samples to collect
        """
        class_dir = self.create_class_directory(gesture_name)
        existing_files = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        collected = 0
        start_collecting = False
        
        print(f"\n{'='*60}")
        print(f"Collecting data for: {gesture_name}")
        print(f"Target samples: {num_samples}")
        print(f"Existing samples: {existing_files}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Press 'S' to START collecting")
        print("- Press 'P' to PAUSE")
        print("- Press 'Q' to QUIT and move to next gesture")
        print("- Hold the gesture steady while collecting")
        print(f"\nGesture Guide for '{gesture_name}':")
        
        # Show gesture instructions
        gesture_instructions = {
            'SPACE': 'Show OPEN PALM facing camera',
            'SEND': 'Show THUMBS UP',
            'BACKSPACE': 'Show THUMBS DOWN',
            'CLEAR': 'Show CLOSED FIST with both hands',
            'MODE_SWITCH': 'Extend PINKY finger (like "hang loose")',
            'PAUSE': 'Show PEACE SIGN (2 fingers)',
        }
        
        print(f"👉 {gesture_instructions.get(gesture_name, 'Follow the gesture')}")
        print()
        
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display info
            status = "COLLECTING" if start_collecting else "PAUSED"
            color = (0, 255, 0) if start_collecting else (0, 0, 255)
            
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Collected: {collected}/{num_samples}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'S' to start, 'P' to pause, 'Q' to quit", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Progress bar
            progress = int((collected / num_samples) * 600)
            cv2.rectangle(frame, (10, 120), (610, 140), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 120), (10 + progress, 140), (0, 255, 0), -1)
            
            cv2.imshow('Data Collection', frame)
            
            # Save frame if collecting
            if start_collecting and results.multi_hand_landmarks:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{gesture_name}_{existing_files + collected}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, frame)
                collected += 1
                time.sleep(0.05)  # Small delay between captures
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                start_collecting = True
                print("▶️  Started collecting...")
            elif key == ord('p') or key == ord('P'):
                start_collecting = False
                print("⏸️  Paused collecting...")
            elif key == ord('q') or key == ord('Q'):
                print(f"⏹️  Stopped. Collected {collected} samples for {gesture_name}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Completed! Total samples for {gesture_name}: {existing_files + collected}")
        time.sleep(1)
    
    def collect_all_control_gestures(self, samples_per_gesture=500):
        """Collect data for all control gestures"""
        print("\n" + "="*70)
        print("CONTROL GESTURE DATA COLLECTION")
        print("="*70)
        print("\nWe will collect data for the following control gestures:")
        for i, gesture in enumerate(config.CONTROLS, 1):
            print(f"{i}. {gesture}")
        
        print("\n⚠️  IMPORTANT:")
        print("- Collect data in different lighting conditions")
        print("- Vary hand positions and angles")
        print("- Use different backgrounds if possible")
        print("- Include both left and right hands (if applicable)")
        print("="*70)
        
        input("\nPress ENTER to start collection...")
        
        for gesture in config.CONTROLS:
            self.collect_gesture_data(gesture, samples_per_gesture)
            
            if gesture != config.CONTROLS[-1]:
                print(f"\n✨ Get ready for next gesture...")
                time.sleep(2)
        
        print("\n" + "="*70)
        print("🎉 DATA COLLECTION COMPLETE!")
        print("="*70)
        print(f"\nData saved to: {self.data_dir}")
        print("\nNext step: Run 'python preprocess_data.py' to prepare data for training")

def main():
    """Main function"""
    print("\n🎥 Sign Language Data Collection Tool")
    print("="*70)
    
    collector = DataCollector()
    
    print("\nOptions:")
    print("1. Collect all control gestures (recommended)")
    print("2. Collect specific gesture")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        samples = input("Enter number of samples per gesture (default: 500): ").strip()
        samples = int(samples) if samples.isdigit() else 500
        collector.collect_all_control_gestures(samples)
    elif choice == '2':
        print("\nAvailable gestures:")
        for i, gesture in enumerate(config.CONTROLS, 1):
            print(f"{i}. {gesture}")
        
        gesture_choice = input("\nEnter gesture number: ").strip()
        if gesture_choice.isdigit() and 1 <= int(gesture_choice) <= len(config.CONTROLS):
            gesture = config.CONTROLS[int(gesture_choice) - 1]
            samples = input(f"Enter number of samples for {gesture} (default: 500): ").strip()
            samples = int(samples) if samples.isdigit() else 500
            collector.collect_gesture_data(gesture, samples)
        else:
            print("❌ Invalid choice!")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
