#!/usr/bin/env python3
"""
Video Buffer Capture Implementation (Option 1)
For capturing clear images from moving conveyor when 24V signal is received.
"""

import cv2
import numpy as np
import threading
import time
import os
import tempfile
from collections import deque
import queue

class VideoBufferCapture:
    """Video buffer capture system for moving conveyor applications"""
    
    def __init__(self, buffer_duration=5.0, frame_rate=30, max_buffer_size=150):
        """
        Initialize video buffer capture
        
        Args:
            buffer_duration: Duration of video buffer in seconds
            frame_rate: Target frame rate for buffer
            max_buffer_size: Maximum number of frames to keep in buffer
        """
        self.buffer_duration = buffer_duration
        self.frame_rate = frame_rate
        self.max_buffer_size = max_buffer_size
        
        # Video buffer (circular buffer of frames with timestamps)
        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.camera = None
        
        # Signal processing
        self.signal_received = False
        self.signal_timestamp = None
        self.processing_lock = threading.Lock()
        
        # Temporary file management
        self.temp_files = []
        
    def start_recording(self, camera_url):
        """Start video buffer recording"""
        if self.is_recording:
            print("Video buffer recording already active")
            return False
        try:
            # Use GStreamer pipeline for RTSP streams
            if camera_url.startswith("rtsp://") or camera_url.startswith("http://"):
                gst = (
                    f"rtspsrc location={camera_url} latency=200 ! "
                    "rtph264depay ! h264parse ! avdec_h264 ! "
                    "nvvidconv ! video/x-raw,format=BGRx ! "
                    "videoconvert ! appsink"
                )
                self.camera = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            else:
                self.camera = cv2.VideoCapture(camera_url)
            if not self.camera.isOpened():
                print(f"Failed to open camera: {camera_url}")
                return False
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
            self.recording_thread.start()
            print(f"Video buffer recording started - Buffer: {self.buffer_duration}s, FPS: {self.frame_rate}")
            return True
        except Exception as e:
            print(f"Error starting video buffer recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop video buffer recording"""
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        # Clear buffer
        with self.buffer_lock:
            self.frame_buffer.clear()
            
        print("Video buffer recording stopped")
    
    def _recording_worker(self):
        """Worker thread for continuous video buffer recording"""
        frame_interval = 1.0 / self.frame_rate
        last_frame_time = 0
        while self.is_recording and self.camera:
            try:
                current_time = time.time()
                # Maintain frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small delay
                    continue
                # Capture frame
                ret, frame = self.camera.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[VideoBuffer] Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    # Add frame to buffer with timestamp
                    with self.buffer_lock:
                        self.frame_buffer.append({
                            'frame': frame.copy(),  # Make a copy to avoid reference issues
                            'timestamp': current_time
                        })
                    last_frame_time = current_time
                else:
                    print("Failed to capture frame or empty frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in recording worker: {e}")
                time.sleep(0.1)
    
    def on_signal_received(self):
        """Called when 24V signal is received"""
        with self.processing_lock:
            self.signal_received = True
            self.signal_timestamp = time.time()
            print(f"24V Signal received at {self.signal_timestamp}")
    
    def extract_best_frame(self, delay_before_signal=1.5):
        """
        Extract the best frame from video buffer when signal is received
        
        Args:
            delay_before_signal: Time before signal to extract frame (seconds)
            
        Returns:
            tuple: (frame, timestamp) or (None, None) if no suitable frame found
        """
        if not self.signal_received:
            print("No signal received yet")
            return None, None
            
        with self.processing_lock:
            if not self.signal_timestamp:
                return None, None
                
            target_time = self.signal_timestamp - delay_before_signal
            
            with self.buffer_lock:
                if not self.frame_buffer:
                    print("No frames in buffer")
                    return None, None
                
                # Find the frame closest to target time
                best_frame = None
                best_timestamp = None
                min_time_diff = float('inf')
                
                for frame_data in self.frame_buffer:
                    time_diff = abs(frame_data['timestamp'] - target_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_frame = frame_data['frame']
                        best_timestamp = frame_data['timestamp']
                
                if best_frame is not None:
                    print(f"Extracted frame from {best_timestamp:.2f}s (target: {target_time:.2f}s)")
                    return best_frame, best_timestamp
                else:
                    print("No suitable frame found in buffer")
                    return None, None
    
    def process_signal_capture(self, inference_model, delay_before_signal=1.5):
        """
        Process signal-triggered capture with video buffer
        
        Args:
            inference_model: Model for quality analysis
            delay_before_signal: Time before signal to extract frame (seconds)
            
        Returns:
            dict: Analysis results or None if failed
        """
        if not self.signal_received:
            print("No signal received for processing")
            return None
            
        try:
            # Extract best frame from buffer
            frame, timestamp = self.extract_best_frame(delay_before_signal)
            if frame is None:
                print("Failed to extract frame from buffer")
                return None
            
            # Create temporary file for processing
            temp_file = self._create_temp_file(frame)
            if not temp_file:
                print("Failed to create temporary file")
                return None
            
            try:
                # Run inference
                results, orig_image = inference_model.predict_single(temp_file)
                print(f"Signal processing completed - Quality: {results['overall_quality']}")
                return results
                
            finally:
                # Clean up temporary file
                self._cleanup_temp_file(temp_file)
                
        except Exception as e:
            print(f"Error in signal processing: {e}")
            return None
    
    def _create_temp_file(self, frame):
        """Create temporary file for frame processing"""
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix='buffer_frame_')
            os.close(temp_fd)  # Close the file descriptor
            
            # Save frame to temporary file
            success = cv2.imwrite(temp_path, frame)
            if success:
                self.temp_files.append(temp_path)
                return temp_path
            else:
                print("Failed to save frame to temporary file")
                return None
                
        except Exception as e:
            print(f"Error creating temporary file: {e}")
            return None
    
    def _cleanup_temp_file(self, temp_path):
        """Clean up temporary file"""
        try:
            if temp_path in self.temp_files:
                self.temp_files.remove(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
                
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")
    
    def cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        for temp_path in self.temp_files[:]:  # Copy list to avoid modification during iteration
            self._cleanup_temp_file(temp_path)
    
    def get_buffer_info(self):
        """Get information about current buffer state"""
        with self.buffer_lock:
            buffer_size = len(self.frame_buffer)
            if buffer_size > 0:
                oldest_time = self.frame_buffer[0]['timestamp']
                newest_time = self.frame_buffer[-1]['timestamp']
                buffer_duration = newest_time - oldest_time
            else:
                buffer_duration = 0
                
        return {
            'buffer_size': buffer_size,
            'buffer_duration': buffer_duration,
            'is_recording': self.is_recording,
            'signal_received': self.signal_received,
            'temp_files_count': len(self.temp_files)
        }
    
    def reset_signal_state(self):
        """Reset signal state for next capture"""
        with self.processing_lock:
            self.signal_received = False
            self.signal_timestamp = None
        print("Signal state reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_recording()
        self.cleanup_all_temp_files() 