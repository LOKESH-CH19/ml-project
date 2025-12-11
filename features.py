"""
Quick Phrases System
Manage and access frequently used phrases
"""

import json
import os
import config

class QuickPhrasesManager:
    """Manage quick phrases for fast communication"""
    
    def __init__(self, phrases_file=None):
        self.phrases_file = phrases_file or config.PHRASES_FILE
        self.phrases = self.load_phrases()
    
    def load_phrases(self):
        """Load phrases from file or create default"""
        if os.path.exists(self.phrases_file):
            try:
                with open(self.phrases_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading phrases: {e}")
                return self.get_default_phrases()
        else:
            phrases = self.get_default_phrases()
            self.save_phrases(phrases)
            return phrases
    
    def get_default_phrases(self):
        """Get default quick phrases"""
        return {
            '1': config.DEFAULT_QUICK_PHRASES[0],
            '2': config.DEFAULT_QUICK_PHRASES[1],
            '3': config.DEFAULT_QUICK_PHRASES[2],
            '4': config.DEFAULT_QUICK_PHRASES[3],
            '5': config.DEFAULT_QUICK_PHRASES[4],
            '6': config.DEFAULT_QUICK_PHRASES[5],
            '7': config.DEFAULT_QUICK_PHRASES[6],
            '8': config.DEFAULT_QUICK_PHRASES[7],
            '9': config.DEFAULT_QUICK_PHRASES[8],
            '0': config.DEFAULT_QUICK_PHRASES[9],
        }
    
    def save_phrases(self, phrases=None):
        """Save phrases to file"""
        if phrases is None:
            phrases = self.phrases
        
        try:
            with open(self.phrases_file, 'w', encoding='utf-8') as f:
                json.dump(phrases, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving phrases: {e}")
            return False
    
    def get_phrase(self, number):
        """Get phrase by number (0-9)"""
        return self.phrases.get(str(number), None)
    
    def set_phrase(self, number, phrase):
        """Set phrase for a number"""
        self.phrases[str(number)] = phrase
        self.save_phrases()
    
    def get_all_phrases(self):
        """Get all phrases"""
        return self.phrases
    
    def delete_phrase(self, number):
        """Delete a phrase"""
        if str(number) in self.phrases:
            del self.phrases[str(number)]
            self.save_phrases()
            return True
        return False
    
    def list_phrases(self):
        """List all phrases"""
        print("\n📝 Quick Phrases:")
        print("="*60)
        for num in sorted(self.phrases.keys()):
            phrase = self.phrases[num]
            print(f"{num}: {phrase}")
        print("="*60)

class ContactManager:
    """Manage contacts for quick communication"""
    
    def __init__(self, contacts_file=None):
        self.contacts_file = contacts_file or config.CONTACTS_FILE
        self.contacts = self.load_contacts()
    
    def load_contacts(self):
        """Load contacts from file"""
        if os.path.exists(self.contacts_file):
            try:
                with open(self.contacts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading contacts: {e}")
                return {}
        else:
            return {}
    
    def save_contacts(self):
        """Save contacts to file"""
        try:
            with open(self.contacts_file, 'w', encoding='utf-8') as f:
                json.dump(self.contacts, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving contacts: {e}")
            return False
    
    def add_contact(self, number, name, phone=None, email=None):
        """
        Add a contact
        
        Args:
            number: Quick access number (1-20)
            name: Contact name
            phone: Phone number (optional)
            email: Email address (optional)
        """
        if len(self.contacts) >= config.MAX_CONTACTS:
            return False, "Maximum contacts reached"
        
        self.contacts[str(number)] = {
            'name': name,
            'phone': phone,
            'email': email
        }
        
        self.save_contacts()
        return True, f"Contact '{name}' added"
    
    def get_contact(self, number):
        """Get contact by number"""
        return self.contacts.get(str(number), None)
    
    def delete_contact(self, number):
        """Delete a contact"""
        if str(number) in self.contacts:
            name = self.contacts[str(number)]['name']
            del self.contacts[str(number)]
            self.save_contacts()
            return True, f"Contact '{name}' deleted"
        return False, "Contact not found"
    
    def list_contacts(self):
        """List all contacts"""
        print("\n📞 Contacts:")
        print("="*60)
        if not self.contacts:
            print("No contacts saved")
        else:
            for num in sorted(self.contacts.keys(), key=int):
                contact = self.contacts[num]
                print(f"{num}: {contact['name']}")
                if contact.get('phone'):
                    print(f"    📱 {contact['phone']}")
                if contact.get('email'):
                    print(f"    ✉️  {contact['email']}")
        print("="*60)
    
    def get_all_contacts(self):
        """Get all contacts"""
        return self.contacts

class ConversationHistory:
    """Manage conversation history"""
    
    def __init__(self, history_file=None):
        self.history_file = history_file or config.HISTORY_FILE
        self.history = self.load_history()
    
    def load_history(self):
        """Load history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        else:
            return []
    
    def save_history(self):
        """Save history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def add_message(self, message, timestamp=None):
        """Add a message to history"""
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = {
            'message': message,
            'timestamp': timestamp
        }
        
        self.history.append(entry)
        
        # Keep only last MAX_HISTORY messages
        if len(self.history) > config.MAX_HISTORY:
            self.history = self.history[-config.MAX_HISTORY:]
        
        self.save_history()
    
    def get_recent(self, count=10):
        """Get recent messages"""
        return self.history[-count:]
    
    def clear_history(self):
        """Clear all history"""
        self.history = []
        self.save_history()
    
    def list_history(self, count=10):
        """List recent history"""
        print("\n💬 Recent Messages:")
        print("="*60)
        recent = self.get_recent(count)
        if not recent:
            print("No messages in history")
        else:
            for entry in recent:
                print(f"[{entry['timestamp']}] {entry['message']}")
        print("="*60)

if __name__ == "__main__":
    # Test quick phrases
    print("Testing Quick Phrases System\n")
    
    phrases_mgr = QuickPhrasesManager()
    phrases_mgr.list_phrases()
    
    # Test contacts
    print("\n" + "="*60)
    contacts_mgr = ContactManager()
    contacts_mgr.add_contact(1, "John Doe", "+1234567890", "john@example.com")
    contacts_mgr.add_contact(2, "Jane Smith", "+0987654321")
    contacts_mgr.list_contacts()
    
    # Test history
    print("\n" + "="*60)
    history = ConversationHistory()
    history.add_message("Hello, how are you?")
    history.add_message("I need help")
    history.list_history()
