#!/usr/bin/env python3
"""
Configuration manager for the Battery Annotation system
"""

import json
import os
from pathlib import Path

class ConfigManager:
    """Manages application configuration settings"""
    
    def __init__(self, config_file="settings.json"):
        self.config_file = config_file
        self.settings = self._load_settings()
    
    def _load_settings(self):
        """Load settings from JSON file or create default settings"""
        default_settings = {
            "com_port": None,  # Will be auto-detected
            "baud_rate": 19200,
            "modbus_slave_id": 1,
            "camera_url": "http://192.168.100.50:8080/stream-hd",
            "capture_interval": 5.0
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults
                    default_settings.update(loaded_settings)
        except Exception as e:
            print(f"Warning: Could not load settings from {self.config_file}: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings to {self.config_file}: {e}")
    
    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set a setting value"""
        self.settings[key] = value
        self.save_settings() 