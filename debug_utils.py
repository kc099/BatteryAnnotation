#!/usr/bin/env python3
"""
Debug utilities for the Battery Annotation system
"""

import os
import sys
from datetime import datetime

# Global debug state
DEBUG_ENABLED = True

def debug_print(message, category="info"):
    """Print debug message with timestamp and category"""
    if not DEBUG_ENABLED:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    category_prefix = {
        "info": "[INFO]",
        "signal": "[SIGNAL]",
        "modbus": "[MODBUS]",
        "errors": "[ERROR]",
        "warning": "[WARN]",
        "success": "[SUCCESS]"
    }.get(category, "[INFO]")
    
    print(f"[{timestamp}] {category_prefix} {message}")

def enable_debug_mode():
    """Enable debug mode"""
    global DEBUG_ENABLED
    DEBUG_ENABLED = True
    debug_print("Debug mode enabled", "info")

def disable_debug_mode():
    """Disable debug mode"""
    global DEBUG_ENABLED
    DEBUG_ENABLED = False
    debug_print("Debug mode disabled", "info")

def toggle_debug_mode():
    """Toggle debug mode"""
    global DEBUG_ENABLED
    DEBUG_ENABLED = not DEBUG_ENABLED
    status = "enabled" if DEBUG_ENABLED else "disabled"
    debug_print(f"Debug mode {status}", "info")

def debug_on():
    """Alias for enable_debug_mode"""
    enable_debug_mode()

def debug_off():
    """Alias for disable_debug_mode"""
    disable_debug_mode()

def debug_toggle():
    """Alias for toggle_debug_mode"""
    toggle_debug_mode() 