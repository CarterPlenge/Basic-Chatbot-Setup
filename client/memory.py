from typing import List, Dict, Any
from pathlib import Path
import json
import time

class Memory:
    """Conversation memory management"""
    
    def __init__(self, memory_file: str = "conversation_memory.json", max_messages: int = 50):
        self.conversations: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self.memory_file = Path(memory_file)
        self.load_memory()
    
    def load_memory(self):
        """Load conversation history from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    self.conversations = json.load(f)
                print(f"Loaded {len(self.conversations)} messages from memory")
        except Exception as e:
            print(f"Failed to load memory: {e}")
            self.conversations = []
    
    def save_memory(self):
        """Save conversation history to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            print(f"Failed to save memory: {e}")
    
    def add_message(self, context_id:str, role: str, content: str):
        """Add a message to conversation history"""
        message = {
            "context_id": context_id,
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        self.conversations.append(message)
        
        # Keep only recent messages
        if len(self.conversations) > self.max_messages:
            self.conversations = self.conversations[-self.max_messages:]
        
        self.save_memory()
    
    def get_conversation_context(self, context_id:str) -> List[Dict[str, str]]:
        """Get conversation context for AI model"""
        return [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in self.conversations 
            if msg["context_id"] == context_id
            ]
    
    def clear_memory(self, context_id):
        """Clear all conversation history"""
        if context_id is None:
            self.conversations = []
            self.save_memory()
            print("All memory cleared")
        else:
            self.conversations = [x for x in self.conversations if x["context_id"] != context_id]
            self.save_memory()
            print(f"Memory cleared for context id: {context_id}")
