"""
Smart Typing Engine
Handles gesture detection, hold detection, cooldown, auto-capitalization
"""

import time
import numpy as np
from collections import deque
import config

class SmartTypingEngine:
    """Smart typing features for sign language detection"""
    
    def __init__(self):
        self.text_buffer = ""
        self.current_mode = 'LETTER'  # LETTER, NUMBER, or CONTROL
        self.last_detection_time = 0
        self.gesture_start_time = None
        self.current_gesture = None
        self.gesture_history = deque(maxlen=10)
        self.sentence_started = False
        
        # Cooldown and hold settings
        self.hold_time = config.HOLD_TIME
        self.cooldown_time = config.COOLDOWN_TIME
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # State tracking
        self.is_in_cooldown = False
        self.is_holding = False
    
    def reset(self):
        """Reset the typing engine"""
        self.text_buffer = ""
        self.sentence_started = False
        self.gesture_history.clear()
        self.current_gesture = None
        self.gesture_start_time = None
    
    def switch_mode(self):
        """Switch between LETTER and NUMBER modes"""
        if self.current_mode == 'LETTER':
            self.current_mode = 'NUMBER'
        else:
            self.current_mode = 'LETTER'
        
        return self.current_mode
    
    def is_cooldown_active(self):
        """Check if cooldown period is active"""
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time
        return time_since_last < self.cooldown_time
    
    def update_gesture_hold(self, detected_gesture, confidence):
        """
        Update gesture hold tracking
        
        Returns:
            True if gesture has been held long enough, False otherwise
        """
        current_time = time.time()
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.gesture_start_time = None
            self.current_gesture = None
            return False
        
        # Check cooldown
        if self.is_cooldown_active():
            return False
        
        # New gesture detected
        if detected_gesture != self.current_gesture:
            self.current_gesture = detected_gesture
            self.gesture_start_time = current_time
            return False
        
        # Same gesture - check hold time
        if self.gesture_start_time is not None:
            hold_duration = current_time - self.gesture_start_time
            
            if hold_duration >= self.hold_time:
                # Gesture held long enough!
                self.last_detection_time = current_time
                self.gesture_start_time = None
                return True
        
        return False
    
    def get_hold_progress(self):
        """Get progress of current gesture hold (0.0 to 1.0)"""
        if self.gesture_start_time is None:
            return 0.0
        
        current_time = time.time()
        hold_duration = current_time - self.gesture_start_time
        progress = min(hold_duration / self.hold_time, 1.0)
        
        return progress
    
    def auto_capitalize(self, char):
        """Apply auto-capitalization rules"""
        # Capitalize first letter of sentence
        if not self.sentence_started or self.text_buffer.endswith('. ') or \
           self.text_buffer.endswith('? ') or self.text_buffer.endswith('! '):
            return char.upper()
        
        return char.lower()
    
    def process_letter(self, letter):
        """Process a detected letter"""
        if letter in config.LETTERS:
            # Apply auto-capitalization
            processed_letter = self.auto_capitalize(letter)
            self.text_buffer += processed_letter
            self.sentence_started = True
            self.gesture_history.append(('LETTER', letter))
            return True
        return False
    
    def process_number(self, number):
        """Process a detected number"""
        if number in config.NUMBERS:
            self.text_buffer += number
            self.gesture_history.append(('NUMBER', number))
            return True
        return False
    
    def process_control(self, control):
        """
        Process a control gesture (mapped from dataset gestures)
        
        Returns:
            Tuple (success, action) where action is the control performed
        """
        # Map dataset gesture names to control actions
        action = config.CONTROL_MAPPINGS.get(control, control)
        
        if action == 'SPACE':
            self.text_buffer += ' '
            self.gesture_history.append(('CONTROL', 'SPACE'))
            return True, 'SPACE'
        
        elif action == 'BACKSPACE':
            if len(self.text_buffer) > 0:
                self.text_buffer = self.text_buffer[:-1]
                self.gesture_history.append(('CONTROL', 'BACKSPACE'))
                return True, 'BACKSPACE'
            return False, 'BACKSPACE'
        
        elif action == 'SEND':
            return True, 'SEND'
        
        elif action == 'MODE_SWITCH':
            # Switch between LETTER and NUMBER modes
            new_mode = self.switch_mode()
            self.gesture_history.append(('CONTROL', 'MODE_SWITCH'))
            return True, f'MODE_SWITCH_{new_mode}'
        
        return False, None
    
    def process_detection(self, detected_class, confidence):
        """
        Process a detection and update text buffer
        
        Args:
            detected_class: The detected class name
            confidence: Confidence score (0-1)
        
        Returns:
            Dictionary with processing results
        """
        result = {
            'success': False,
            'action': None,
            'text': self.text_buffer,
            'mode': self.current_mode,
            'hold_progress': 0.0
        }
        
        # Check if gesture should be processed (hold check)
        if not self.update_gesture_hold(detected_class, confidence):
            result['hold_progress'] = self.get_hold_progress()
            return result
        
        # Check if Y gesture is used for mode switching (prioritize over letter Y)
        if detected_class == 'Y':
            success, action = self.process_control('Y')
            result['success'] = success
            result['action'] = action
        
        # Process based on mode and class
        elif detected_class in config.CONTROLS:
            success, action = self.process_control(detected_class)
            result['success'] = success
            result['action'] = action
        
        elif self.current_mode == 'LETTER' and detected_class in config.LETTERS:
            result['success'] = self.process_letter(detected_class)
            result['action'] = 'ADD_LETTER'
        
        elif self.current_mode == 'NUMBER' and detected_class in config.NUMBERS:
            result['success'] = self.process_number(detected_class)
            result['action'] = 'ADD_NUMBER'
        
        result['text'] = self.text_buffer
        
        return result
    
    def get_text(self):
        """Get current text buffer"""
        return self.text_buffer
    
    def set_text(self, text):
        """Set text buffer"""
        self.text_buffer = text
    
    def get_last_word(self):
        """Get the last word in text buffer"""
        words = self.text_buffer.strip().split()
        return words[-1] if words else ""
    
    def suggest_completions(self, word_list):
        """
        Suggest word completions based on current partial word
        
        Args:
            word_list: List of common words
        
        Returns:
            List of suggested completions
        """
        last_word = self.get_last_word().lower()
        
        if not last_word:
            return []
        
        suggestions = [word for word in word_list if word.lower().startswith(last_word)]
        return suggestions[:5]  # Return top 5 suggestions

if __name__ == "__main__":
    # Test smart typing engine
    engine = SmartTypingEngine()
    
    print("Testing Smart Typing Engine\n")
    print("="*50)
    
    # Simulate detections
    test_sequence = [
        ('H', 0.95, 'LETTER'),
        ('E', 0.92, 'LETTER'),
        ('L', 0.88, 'LETTER'),
        ('L', 0.90, 'LETTER'),
        ('O', 0.93, 'LETTER'),
        ('SPACE', 0.96, 'CONTROL'),
        ('MODE_SWITCH', 0.94, 'CONTROL'),
        ('5', 0.91, 'NUMBER'),
    ]
    
    for char, conf, expected_mode in test_sequence:
        time.sleep(1.6)  # Simulate hold time
        result = engine.process_detection(char, conf)
        print(f"Detected: {char} | Confidence: {conf} | Success: {result['success']}")
        print(f"Text: '{result['text']}' | Mode: {result['mode']}")
        print("-"*50)
    
    print(f"\nFinal text: '{engine.get_text()}'")
