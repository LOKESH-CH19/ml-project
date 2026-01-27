"""
Real-time Sign Language Detection Application
Main GUI application with all features
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import pyttsx3
import threading
import config
from smart_typing import SmartTypingEngine
from features import QuickPhrasesManager, ContactManager, ConversationHistory

class SignLanguageApp:
    """Main application for real-time sign language detection"""
    
    def __init__(self, model_path):
        # Initialize window
        self.root = tk.Tk()
        self.root.title("Sign Language Detection App")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg='#2C3E50')
        
        # Load model
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class names
        with open(os.path.join(config.PROCESSED_DATA_DIR, 'class_names.pkl'), 'rb') as f:
            self.class_names = pickle.load(f)
        
        print(f"Model loaded! Classes: {len(self.class_names)}")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Initialize features
        self.typing_engine = SmartTypingEngine()
        self.phrases_manager = QuickPhrasesManager()
        self.contacts_manager = ContactManager()
        self.history = ConversationHistory()
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', config.TTS_RATE)
        self.tts_engine.setProperty('volume', config.TTS_VOLUME)
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        # State variables
        self.is_running = True
        self.is_paused = False
        self.last_prediction = None
        self.last_confidence = 0
        
        # Create GUI
        self.create_gui()
        
        # Start video loop
        self.update_frame()
    
    def create_gui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_frame, bg='#34495E', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video label
        self.video_label = tk.Label(left_panel, bg='black')
        self.video_label.pack(padx=10, pady=10)
        
        # Detection info frame
        info_frame = tk.Frame(left_panel, bg='#34495E')
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current detection
        self.detection_label = tk.Label(
            info_frame, text="Detected: None", 
            font=('Arial', 14, 'bold'), bg='#34495E', fg='#ECF0F1'
        )
        self.detection_label.pack()
        
        # Current gesture preview (large)
        self.gesture_preview = tk.Label(
            info_frame, text="👋",
            font=('Arial', 72), bg='#34495E', fg='#3498DB'
        )
        self.gesture_preview.pack(pady=10)
        
        # Confidence bar
        self.confidence_label = tk.Label(
            info_frame, text="Confidence: 0%",
            font=('Arial', 12), bg='#34495E', fg='#ECF0F1'
        )
        self.confidence_label.pack()
        
        self.confidence_bar = ttk.Progressbar(
            info_frame, length=300, mode='determinate'
        )
        self.confidence_bar.pack(pady=5)
        
        # Hold progress bar
        self.hold_label = tk.Label(
            info_frame, text="Hold Progress:",
            font=('Arial', 10), bg='#34495E', fg='#ECF0F1'
        )
        self.hold_label.pack()
        
        self.hold_bar = ttk.Progressbar(
            info_frame, length=300, mode='determinate'
        )
        self.hold_bar.pack(pady=5)
        
        # Right panel - Controls and text
        right_panel = tk.Frame(main_frame, bg='#34495E', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Mode indicator
        mode_frame = tk.Frame(right_panel, bg='#34495E')
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            mode_frame, text="Current Mode:",
            font=('Arial', 12, 'bold'), bg='#34495E', fg='#ECF0F1'
        ).pack(side=tk.LEFT)
        
        self.mode_label = tk.Label(
            mode_frame, text="ABC (Letters)",
            font=('Arial', 12, 'bold'), bg='#3498DB', fg='white',
            padx=10, pady=5, relief=tk.RAISED
        )
        self.mode_label.pack(side=tk.LEFT, padx=10)
        
        # Word predictions
        predictions_frame = tk.Frame(right_panel, bg='#34495E')
        predictions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            predictions_frame, text="Suggestions:",
            font=('Arial', 10, 'bold'), bg='#34495E', fg='#ECF0F1'
        ).pack(anchor=tk.W)
        
        self.suggestions_frame = tk.Frame(predictions_frame, bg='#34495E')
        self.suggestions_frame.pack(fill=tk.X)
        
        self.suggestion_buttons = []
        for i in range(3):
            btn = tk.Button(
                self.suggestions_frame, text="", font=('Arial', 9),
                bg='#3498DB', fg='white', relief=tk.RAISED,
                command=lambda idx=i: self.use_suggestion(idx)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.suggestion_buttons.append(btn)
        
        # Text display
        tk.Label(
            right_panel, text="Message:",
            font=('Arial', 12, 'bold'), bg='#34495E', fg='#ECF0F1'
        ).pack(padx=10, pady=(10, 0))
        
        self.text_display = tk.Text(
            right_panel, height=8, width=40,
            font=('Arial', 14), wrap=tk.WORD,
            bg='#ECF0F1', fg='#2C3E50'
        )
        self.text_display.pack(padx=10, pady=10)
        
        # Control buttons
        btn_frame = tk.Frame(right_panel, bg='#34495E')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.speak_btn = tk.Button(
            btn_frame, text="🔊 Speak", command=self.speak_text,
            font=('Arial', 11, 'bold'), bg='#27AE60', fg='white',
            padx=10, pady=5, relief=tk.RAISED, cursor='hand2'
        )
        self.speak_btn.pack(side=tk.LEFT, padx=2)
        
        self.undo_btn = tk.Button(
            btn_frame, text="↶ Undo", command=self.undo_last,
            font=('Arial', 11, 'bold'), bg='#E67E22', fg='white',
            padx=10, pady=5, relief=tk.RAISED, cursor='hand2'
        )
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        
        self.clear_btn = tk.Button(
            btn_frame, text="🗑️ Clear", command=self.clear_text,
            font=('Arial', 11, 'bold'), bg='#E74C3C', fg='white',
            padx=10, pady=5, relief=tk.RAISED, cursor='hand2'
        )
        self.clear_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = tk.Button(
            btn_frame, text="⏸️ Pause", command=self.toggle_pause,
            font=('Arial', 11, 'bold'), bg='#F39C12', fg='white',
            padx=10, pady=5, relief=tk.RAISED, cursor='hand2'
        )
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        # Quick phrases section
        tk.Label(
            right_panel, text="Quick Phrases (0-9):",
            font=('Arial', 11, 'bold'), bg='#34495E', fg='#ECF0F1'
        ).pack(padx=10, pady=(10, 5))
        
        phrases_frame = tk.Frame(right_panel, bg='#34495E')
        phrases_frame.pack(fill=tk.X, padx=10)
        
        self.phrase_listbox = tk.Listbox(
            phrases_frame, height=6, font=('Arial', 9),
            bg='#ECF0F1', fg='#2C3E50'
        )
        self.phrase_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.phrase_listbox.bind('<Double-Button-1>', self.use_quick_phrase)
        
        phrase_scroll = tk.Scrollbar(phrases_frame, command=self.phrase_listbox.yview)
        phrase_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.phrase_listbox.config(yscrollcommand=phrase_scroll.set)
        
        # Load quick phrases
        self.load_quick_phrases()
        
        # Menu buttons
        menu_frame = tk.Frame(right_panel, bg='#34495E')
        menu_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            menu_frame, text="📝 Manage Phrases", command=self.manage_phrases,
            font=('Arial', 9), bg='#9B59B6', fg='white', cursor='hand2'
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            menu_frame, text="📞 Contacts", command=self.manage_contacts,
            font=('Arial', 9), bg='#16A085', fg='white', cursor='hand2'
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            menu_frame, text="💬 History", command=self.show_history,
            font=('Arial', 9), bg='#34495E', fg='white', cursor='hand2'
        ).pack(side=tk.LEFT, padx=2)
        
        # Status bar
        self.status_label = tk.Label(
            self.root, text="Ready", font=('Arial', 9),
            bg='#1ABC9C', fg='white', anchor=tk.W, padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_quick_phrases(self):
        """Load quick phrases into listbox"""
        self.phrase_listbox.delete(0, tk.END)
        phrases = self.phrases_manager.get_all_phrases()
        for num in sorted(phrases.keys(), key=int):
            self.phrase_listbox.insert(tk.END, f"{num}: {phrases[num]}")
    
    def preprocess_frame(self, frame):
        """Extract hand landmarks from frame for model prediction"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Extract landmarks from first hand
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks[:1]:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Convert to numpy array with batch dimension
            landmarks_array = np.array(landmarks, dtype=np.float32)
            return np.expand_dims(landmarks_array, axis=0)
        
        return None
    
    def predict_sign(self, frame):
        """Predict sign language gesture from frame"""
        processed = self.preprocess_frame(frame)
        
        if processed is None:
            # No hand detected
            return None, 0.0, None
        
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]
        predicted_class = self.class_names[top_idx]
        
        return predicted_class, confidence, predictions
    
    def update_frame(self):
        """Update video frame and process detection"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.after_frame = self.root.after(10, self.update_frame)
            return
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            
            # Make prediction if not paused
            if not self.is_paused:
                predicted_class, confidence, _ = self.predict_sign(frame)
                
                # Only process if hand was detected
                if predicted_class is not None:
                    self.last_prediction = predicted_class
                    self.last_confidence = confidence
                    
                    # Process with typing engine
                    result = self.typing_engine.process_detection(predicted_class, confidence)
                    
                    # Update text display
                    self.text_display.delete('1.0', tk.END)
                    self.text_display.insert('1.0', result['text'])
                    
                    # Update word predictions
                    self.update_word_suggestions(result['text'])
                    
                    # Update gesture preview
                    self.gesture_preview.config(text=predicted_class, fg='#27AE60')
                    
                    # Update mode display
                    mode_text = "ABC (Letters)" if result['mode'] == 'LETTER' else "123 (Numbers)"
                    mode_color = '#3498DB' if result['mode'] == 'LETTER' else '#E67E22'
                    self.mode_label.config(text=mode_text, bg=mode_color)
                    
                    # Update hold progress
                    self.hold_bar['value'] = result['hold_progress'] * 100
                    
                    # Handle actions
                    if result['success'] and result['action'] == 'SEND':
                        self.speak_text()
                    
                    # Update status
                    status_color = '#27AE60' if confidence > config.CONFIDENCE_THRESHOLD else '#E74C3C'
                    self.status_label.config(
                    text=f"Detected: {predicted_class} | Confidence: {confidence:.2%}",
                    bg=status_color
                )
                else:
                    self.gesture_preview.config(text="🤚", fg='#95A5A6')
        else:
            self.last_prediction = None
            self.last_confidence = 0
            self.status_label.config(text="No hand detected", bg='#95A5A6')
        
        # Update detection info
        if self.last_prediction:
            self.detection_label.config(text=f"Detected: {self.last_prediction}")
            self.confidence_label.config(text=f"Confidence: {self.last_confidence:.1%}")
            self.confidence_bar['value'] = self.last_confidence * 100
        else:
            self.detection_label.config(text="Detected: None")
            self.confidence_label.config(text="Confidence: 0%")
            self.confidence_bar['value'] = 0
        
        # Convert frame for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        photo = ImageTk.PhotoImage(image=img)
        
        self.video_label.config(image=photo)
        self.video_label.image = photo
        
        # Schedule next update
        self.after_frame = self.root.after(10, self.update_frame)
    
    def speak_text(self):
        """Speak the current text using TTS"""
        text = self.text_display.get('1.0', tk.END).strip()
        if text:
            # Save to history
            self.history.add_message(text)
            
            # Speak in separate thread
            def speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
            
            threading.Thread(target=speak, daemon=True).start()
            self.status_label.config(text="Speaking...", bg='#3498DB')
        else:
            messagebox.showwarning("Empty Message", "No text to speak!")
    
    def clear_text(self):
        """Clear the text display"""
        self.text_display.delete('1.0', tk.END)
        self.typing_engine.reset()
        self.status_label.config(text="Text cleared", bg='#E74C3C')
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶️ Resume", bg='#27AE60')
            self.status_label.config(text="Detection paused", bg='#F39C12')
        else:
            self.pause_btn.config(text="⏸️ Pause", bg='#F39C12')
            self.status_label.config(text="Detection resumed", bg='#27AE60')
    
    def use_quick_phrase(self, event):
        """Use a quick phrase from listbox"""
        selection = self.phrase_listbox.curselection()
        if selection:
            phrase_text = self.phrase_listbox.get(selection[0])
            # Extract phrase (after "X: ")
            phrase = phrase_text.split(': ', 1)[1]
            
            current_text = self.text_display.get('1.0', tk.END).strip()
            if current_text:
                new_text = current_text + ' ' + phrase
            else:
                new_text = phrase
            
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', new_text)
            self.typing_engine.set_text(new_text)
    
    def manage_phrases(self):
        """Open phrase management dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Quick Phrases")
        dialog.geometry("500x400")
        dialog.configure(bg='#34495E')
        
        # Instructions
        tk.Label(
            dialog, text="Edit Quick Phrases (0-9):",
            font=('Arial', 12, 'bold'), bg='#34495E', fg='white'
        ).pack(pady=10)
        
        # Phrases frame
        frame = tk.Frame(dialog, bg='#34495E')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        entries = {}
        phrases = self.phrases_manager.get_all_phrases()
        
        for num in sorted(phrases.keys(), key=int):
            row_frame = tk.Frame(frame, bg='#34495E')
            row_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(
                row_frame, text=f"{num}:", width=3,
                font=('Arial', 10, 'bold'), bg='#34495E', fg='white'
            ).pack(side=tk.LEFT)
            
            entry = tk.Entry(row_frame, font=('Arial', 10), width=50)
            entry.insert(0, phrases[num])
            entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            entries[num] = entry
        
        # Save button
        def save_phrases():
            for num, entry in entries.items():
                self.phrases_manager.set_phrase(num, entry.get())
            self.load_quick_phrases()
            messagebox.showinfo("Success", "Phrases saved!")
            dialog.destroy()
        
        tk.Button(
            dialog, text="💾 Save", command=save_phrases,
            font=('Arial', 11, 'bold'), bg='#27AE60', fg='white',
            padx=20, pady=5, cursor='hand2'
        ).pack(pady=10)
    
    def manage_contacts(self):
        """Open contacts management dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Contacts")
        dialog.geometry("500x400")
        dialog.configure(bg='#34495E')
        
        tk.Label(
            dialog, text="Contacts:",
            font=('Arial', 12, 'bold'), bg='#34495E', fg='white'
        ).pack(pady=10)
        
        # Contacts list
        contacts_list = tk.Listbox(dialog, font=('Arial', 10), height=15)
        contacts_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def refresh_list():
            contacts_list.delete(0, tk.END)
            contacts = self.contacts_manager.get_all_contacts()
            for num in sorted(contacts.keys(), key=int):
                contact = contacts[num]
                contacts_list.insert(tk.END, f"{num}: {contact['name']} - {contact.get('phone', 'N/A')}")
        
        refresh_list()
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg='#34495E')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def add_contact():
            num = simpledialog.askstring("Add Contact", "Enter number (1-20):")
            if num:
                name = simpledialog.askstring("Add Contact", "Enter name:")
                if name:
                    phone = simpledialog.askstring("Add Contact", "Enter phone (optional):")
                    success, msg = self.contacts_manager.add_contact(num, name, phone)
                    messagebox.showinfo("Contact", msg)
                    refresh_list()
        
        tk.Button(
            btn_frame, text="➕ Add", command=add_contact,
            font=('Arial', 10), bg='#27AE60', fg='white', cursor='hand2'
        ).pack(side=tk.LEFT, padx=2)
    
    def show_history(self):
        """Show conversation history"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Conversation History")
        dialog.geometry("600x400")
        dialog.configure(bg='#34495E')
        
        tk.Label(
            dialog, text="Recent Messages:",
            font=('Arial', 12, 'bold'), bg='#34495E', fg='white'
        ).pack(pady=10)
        
        text_widget = tk.Text(dialog, font=('Arial', 10), wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        recent = self.history.get_recent(20)
        for entry in recent:
            text_widget.insert(tk.END, f"[{entry['timestamp']}]\n{entry['message']}\n\n")
        
        text_widget.config(state=tk.DISABLED)
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if hasattr(self, 'after_frame'):
            self.root.after_cancel(self.after_frame)
        self.cap.release()
        cv2.destroyAllWindows()
    
    def undo_last(self):
        """Undo last character"""
        current_text = self.text_display.get('1.0', tk.END).strip()
        if current_text:
            new_text = current_text[:-1]
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', new_text)
            self.typing_engine.set_text(new_text)
    
    def update_word_suggestions(self, text):
        """Update word prediction suggestions with comprehensive dictionary"""
        # Comprehensive common words dictionary (300+ words)
        common_words = [
            # Greetings & Courtesy (20)
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank", "you", 
            "please", "sorry", "excuse", "me", "welcome", "pardon",
            "greetings", "howdy", "cheers", "farewell", "later", "goodnight",
            
            # Common verbs (40)
            "are", "is", "am", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should",
            "may", "might", "must", "go", "going", "went", "come", "coming",
            "get", "got", "make", "made", "take", "took", "see", "saw", "know",
            "think", "want", "need", "help", "like", "love", "feel",
            
            # Time & Days (25)
            "time", "today", "tomorrow", "yesterday", "now", "later", "soon",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "morning", "afternoon", "evening", "night", "day", "week", "month",
            "year", "hour", "minute", "second",
            
            # Questions (25)
            "what", "when", "where", "who", "why", "which", "how", "whose",
            "whom", "question", "answer", "ask", "tell", "said", "say",
            "mean", "means", "understand", "explain", "repeat", "again",
            "know", "wondering", "curious", "confused",
            
            # Emotions & States (30)
            "good", "great", "fine", "okay", "bad", "terrible", "happy", "sad",
            "angry", "mad", "glad", "excited", "bored", "tired", "sick",
            "hungry", "thirsty", "full", "empty", "ready", "busy", "free",
            "hot", "cold", "warm", "cool", "better", "worse", "best", "worst",
            
            # People & Relationships (20)
            "person", "people", "family", "friend", "mother", "father", "mom",
            "dad", "brother", "sister", "child", "baby", "adult", "man",
            "woman", "boy", "girl", "doctor", "teacher", "student",
            
            # Places (25)
            "home", "house", "work", "school", "hospital", "store", "shop",
            "restaurant", "cafe", "bank", "office", "room", "bathroom",
            "kitchen", "bedroom", "place", "here", "there", "where",
            "city", "town", "country", "world", "street", "road",
            
            # Common nouns (30)
            "thing", "things", "something", "nothing", "everything", "anything",
            "water", "food", "drink", "eat", "money", "phone", "computer",
            "car", "bus", "train", "bike", "book", "paper", "pen", "pencil",
            "table", "chair", "door", "window", "bed", "tv", "radio",
            "bag", "clothes",
            
            # Actions & Activities (35)
            "call", "text", "message", "talk", "speak", "listen", "hear",
            "watch", "look", "read", "write", "study", "learn", "teach",
            "work", "play", "rest", "sleep", "wake", "eat", "drink",
            "walk", "run", "sit", "stand", "lie", "drive", "ride",
            "buy", "sell", "pay", "give", "receive", "send", "open", "close",
            
            # Numbers & Quantities (15)
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "many", "much", "some", "few", "lot",
            
            # Adjectives (25)
            "big", "small", "large", "little", "tall", "short", "long",
            "new", "old", "young", "fast", "slow", "quick", "easy", "hard",
            "difficult", "simple", "nice", "beautiful", "ugly", "clean",
            "dirty", "right", "wrong", "correct",
            
            # Prepositions & Connectors (20)
            "the", "and", "but", "or", "so", "if", "because", "for", "with",
            "without", "about", "from", "into", "through", "during",
            "before", "after", "between", "among", "while",
            
            # Medical & Health (15)
            "hurt", "pain", "medicine", "pill", "doctor", "nurse", "hospital",
            "sick", "ill", "healthy", "emergency", "urgent", "appointment",
            "prescription", "treatment",
            
            # Technology (15)
            "phone", "call", "text", "email", "internet", "wifi", "computer",
            "laptop", "tablet", "app", "website", "online", "offline",
            "charge", "battery"
        ]
        
        words = text.lower().split()
        if not words:
            for btn in self.suggestion_buttons:
                btn.config(text="", state=tk.DISABLED)
            return
        
        last_word = words[-1] if words else ""
        
        # Find matching words (up to 3)
        suggestions = [w for w in common_words if w.startswith(last_word) and w != last_word][:3]
        
        # Update buttons
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.config(text=suggestions[i], state=tk.NORMAL)
            else:
                btn.config(text="", state=tk.DISABLED)
    
    def use_suggestion(self, idx):
        """Use a word suggestion"""
        btn = self.suggestion_buttons[idx]
        word = btn.config('text')[-1]
        if word:
            current_text = self.text_display.get('1.0', tk.END).strip()
            words = current_text.split()
            if words:
                words[-1] = word
                new_text = ' '.join(words)
            else:
                new_text = word
            
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', new_text)
            self.typing_engine.set_text(new_text)
    
    def update_hold_time(self):
        """Update hold time setting"""
        new_time = self.hold_time_var.get()
        self.typing_engine.hold_time = new_time
        config.HOLD_TIME = new_time
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cleanup()
            self.root.destroy()

def main():
    """Main function"""
    print("\n🚀 Sign Language Detection Application")
    print("="*70)
    
    # Check for model
    models = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.h5')]
    
    if not models:
        print("❌ No trained model found!")
        print("\nPlease train a model first:")
        print("  python train_model.py")
        return
    
    print("\n📦 Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    if len(models) == 1:
        model_path = os.path.join(config.MODELS_DIR, models[0])
    else:
        choice = input("\nSelect model (default: 1): ").strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        model_path = os.path.join(config.MODELS_DIR, models[idx])
    
    print(f"\n✅ Loading model: {os.path.basename(model_path)}")
    
    # Run application
    app = SignLanguageApp(model_path)
    app.run()

if __name__ == "__main__":
    main()
