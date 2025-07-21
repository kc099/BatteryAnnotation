import cv2
import numpy as np
import time
import threading
import queue
import traceback
import os

class CameraStreamer:
    """Base class for camera streaming"""
    def __init__(self):
        self.stop_flag = False
        self.is_streaming = False
        self.current_frame = None
        self.thread = None
    
    def start(self):
        """Start the camera streaming thread"""
        if self.is_streaming:
            print("Stream already running")
            return
        
        self.stop_flag = False
        self.thread = threading.Thread(target=self._stream_thread, daemon=True)
        self.thread.start()
        self.is_streaming = True
    
    def stop(self):
        """Stop the camera streaming thread"""
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.is_streaming = False
    
    def get_frame(self):
        """Get the current frame"""
        return self.current_frame
    
    def _stream_thread(self):
        """Streaming thread to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _stream_thread method")

class IPCamera(CameraStreamer):
    """IP camera streamer"""
    def __init__(self, url, intrinsics=None):
        super().__init__()
        self.url = url
        self.cap = None
        self.intrinsics = intrinsics if intrinsics else {}
    
    def _stream_thread(self):
        """Stream from IP camera"""
        try:
            # Open capture
            self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                print(f"Failed to open IP camera at {self.url}")
                return
            
            print(f"IP camera started: {self.url}")
            
            while not self.stop_flag:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from IP camera")
                    time.sleep(1.0)  # Wait before retry
                    continue
                
                # Store current frame
                self.current_frame = frame
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            print(f"Error in IP camera stream: {e}")
            print(traceback.format_exc())
        finally:
            if self.cap:
                self.cap.release()
            print("IP camera stopped")

def create_ip_camera(url):
    """Factory function to create an IP camera instance"""
    return IPCamera(url)
