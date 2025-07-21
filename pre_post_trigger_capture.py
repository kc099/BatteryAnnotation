#!/usr/bin/env python3
"""
Pre-trigger Buffer with Post-trigger Capture Implementation (Option 4)
Industrial standard approach for capturing complete event sequences.
"""

import cv2
import numpy as np
import threading
import time
import os
import tempfile
from collections import deque
import queue
from pathlib import Path
import shutil, datetime

SAVE_DIR = Path.home() / "Amaron" / "BatteryAnnotation" / "captured_frames"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
class PrePostTriggerCapture:
    """Pre-trigger buffer with post-trigger capture system for industrial vision"""
    
    def __init__(self, pre_trigger_duration=3.0, post_trigger_duration=2.0, frame_rate=30):
        """
        Initialize pre-post trigger capture system
        
        Args:
            pre_trigger_duration: Duration of pre-trigger buffer in seconds
            post_trigger_duration: Duration of post-trigger recording in seconds
            frame_rate: Target frame rate for capture
        """
        self.pre_trigger_duration = pre_trigger_duration
        self.post_trigger_duration = post_trigger_duration
        self.frame_rate = frame_rate
        
        # Pre-trigger buffer (circular buffer)
        self.pre_trigger_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # Post-trigger recording
        self.post_trigger_frames = []
        self.post_trigger_timestamps = []
        self.post_trigger_lock = threading.Lock()
        
        # Recording state
        self.is_recording = False
        self.is_post_trigger_recording = False
        self.recording_thread = None
        self.post_trigger_thread = None
        self.camera = None
        
        # Signal processing
        self.signal_received = False
        self.signal_timestamp = None
        self.processing_lock = threading.Lock()
        
        # Temporary file management
        self.temp_files = []
        
        # Frame quality analysis
        self.quality_metrics = []
        
    def start_recording(self, camera_url):
        """Start pre-trigger buffer recording"""
        if self.is_recording:
            print("Pre-trigger recording already active")
            return False
        try:
            # Use GStreamer pipeline for RTSP streams
            #if camera_url.startswith("rtsp://") or camera_url.startswith("http://"):
             #   gst = (
              #      f"rtspsrc location={camera_url} latency=200 ! "
              #      "rtph264depay ! h264parse ! avdec_h264 ! "
              #      "nvvidconv ! video/x-raw,format=BGRx ! "
               #     "videoconvert ! appsink"
                #)
                #self.camera = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            #else:
            self.camera = cv2.VideoCapture(camera_url)
            if not self.camera.isOpened():
                print(f"Failed to open camera: {camera_url}")
                return False
            # Set camera properties for industrial vision
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Fast exposure for moving objects
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._pre_trigger_worker, daemon=True)
            self.recording_thread.start()
            print(f"Pre-trigger recording started - Pre-trigger: {self.pre_trigger_duration}s, Post-trigger: {self.post_trigger_duration}s, FPS: {self.frame_rate}")
            return True
        except Exception as e:
            print(f"Error starting pre-trigger recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop all recording"""
        self.is_recording = False
        self.is_post_trigger_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
            
        if self.post_trigger_thread:
            self.post_trigger_thread.join(timeout=2.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        # Clear buffers
        with self.buffer_lock:
            self.pre_trigger_buffer.clear()
            
        with self.post_trigger_lock:
            self.post_trigger_frames.clear()
            self.post_trigger_timestamps.clear()
            
        print("All recording stopped")
    
    def _pre_trigger_worker(self):
        """Worker thread for pre-trigger buffer recording"""
        frame_interval = 1.0 / self.frame_rate
        last_frame_time = 0
        max_buffer_size = int(self.pre_trigger_duration * self.frame_rate)
        while self.is_recording and self.camera:
            try:
                current_time = time.time()
                # Maintain frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                # Capture frame
                ret, frame = self.camera.read()
                if ret and frame is not None and frame.size > 0:
                    #print(f"[PreTrigger] Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    # Add frame to pre-trigger buffer
                    with self.buffer_lock:
                        self.pre_trigger_buffer.append({
                            'frame': frame.copy(),
                            'timestamp': current_time
                        })
                        # Maintain buffer size
                        while len(self.pre_trigger_buffer) > max_buffer_size:
                            self.pre_trigger_buffer.popleft()
                    last_frame_time = current_time
                else:
                    print("Failed to capture frame or empty frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in pre-trigger worker: {e}")
                time.sleep(0.1)
    
    def _post_trigger_worker(self):
        """Worker thread for post-trigger recording"""
        frame_interval = 1.0 / self.frame_rate
        last_frame_time = 0
        end_time = time.time() + self.post_trigger_duration
        while self.is_post_trigger_recording and time.time() < end_time and self.camera:
            try:
                current_time = time.time()
                # Maintain frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                # Capture frame
                ret, frame = self.camera.read()
                if ret and frame is not None and frame.size > 0:
                    #print(f"[PostTrigger] Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    # Add frame to post-trigger buffer
                    with self.post_trigger_lock:
                        self.post_trigger_frames.append(frame.copy())
                        self.post_trigger_timestamps.append(current_time)
                    last_frame_time = current_time
                else:
                    print("Failed to capture post-trigger frame or empty frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in post-trigger worker: {e}")
                time.sleep(0.1)
        
        self.is_post_trigger_recording = False
        print(f"Post-trigger recording completed - Captured {len(self.post_trigger_frames)} frames")
    
    def on_signal_received(self):
        """Called when 24V signal is received"""
        with self.processing_lock:
            self.signal_received = True
            self.signal_timestamp = time.time()
            print(f"24V Signal received at {self.signal_timestamp}")
            
            # Start post-trigger recording
            if self.is_recording and not self.is_post_trigger_recording:
                self.is_post_trigger_recording = True
                self.post_trigger_thread = threading.Thread(target=self._post_trigger_worker, daemon=True)
                self.post_trigger_thread.start()
                print("Post-trigger recording started")
    
    def _calculate_frame_quality(self, frame):
        """Calculate frame quality metrics for sharpness assessment"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (sharpness measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast
            contrast = gray.std()
            
            # Calculate brightness
            brightness = gray.mean()
            
            return {
                'sharpness': sharpness,
                'contrast': contrast,
                'brightness': brightness,
                'overall_score': sharpness * 0.6 + contrast * 0.3 + (255 - abs(brightness - 128)) * 0.1
            }
        except Exception as e:
            print(f"Error calculating frame quality: {e}")
            return {'overall_score': 0}
    
    def select_best_frame(self, analysis_window=2.0):
        """
        Select the best frame from the complete sequence
        
        Args:
            analysis_window: Time window around signal for analysis (seconds)
            
        Returns:
            tuple: (frame, timestamp, quality_score) or (None, None, None)
        """
        if not self.signal_received or not self.signal_timestamp:
            print("No signal received for frame selection")
            return None, None, None
        
        try:
            # Combine pre-trigger and post-trigger frames
            all_frames = []
            
            # Add pre-trigger frames
            with self.buffer_lock:
                for frame_data in self.pre_trigger_buffer:
                    all_frames.append({
                        'frame': frame_data['frame'],
                        'timestamp': frame_data['timestamp'],
                        'type': 'pre_trigger'
                    })
            
            # Add post-trigger frames
            with self.post_trigger_lock:
                for i, (frame, timestamp) in enumerate(zip(self.post_trigger_frames, self.post_trigger_timestamps)):
                    all_frames.append({
                        'frame': frame,
                        'timestamp': timestamp,
                        'type': 'post_trigger'
                    })
            
            if not all_frames:
                print("No frames available for analysis")
                return None, None, None
            
            # Filter frames within analysis window
            window_start = self.signal_timestamp - analysis_window
            window_end = self.signal_timestamp + analysis_window
            
            candidate_frames = []
            for frame_data in all_frames:
                if window_start <= frame_data['timestamp'] <= window_end:
                    # Calculate quality metrics
                    quality = self._calculate_frame_quality(frame_data['frame'])
                    candidate_frames.append({
                        'frame': frame_data['frame'],
                        'timestamp': frame_data['timestamp'],
                        'type': frame_data['type'],
                        'quality': quality
                    })
            
            if not candidate_frames:
                print("No frames within analysis window")
                return None, None, None
            
            # Select frame with highest quality score
            best_frame_data = max(candidate_frames, key=lambda x: x['quality']['overall_score'])
            
            print(f"Selected best frame: {best_frame_data['timestamp']:.2f}s ({best_frame_data['type']}), "
                  f"Quality score: {best_frame_data['quality']['overall_score']:.2f}")
            
            return best_frame_data['frame'], best_frame_data['timestamp'], best_frame_data['quality']
            
        except Exception as e:
            print(f"Error selecting best frame: {e}")
            return None, None, None
    
    def process_signal_capture(self, inference_model, analysis_window=2.0):
        """
        Process signal-triggered capture with pre-post trigger analysis
        
        Args:
            inference_model: Model for quality analysis
            analysis_window: Time window around signal for analysis (seconds)
            
        Returns:
            dict: Analysis results or None if failed
        """
        if not self.signal_received:
            print("No signal received for processing")
            return None
        
        # Wait for post-trigger recording to complete
        if self.is_post_trigger_recording:
            print("Waiting for post-trigger recording to complete...")
            timeout = time.time() + self.post_trigger_duration + 1.0
            while self.is_post_trigger_recording and time.time() < timeout:
                time.sleep(0.1)
        
        try:
            # Select best frame from complete sequence
            frame, timestamp, quality = self.select_best_frame(analysis_window)
            if frame is None:
                print("Failed to select best frame")
                return None
            
            # Create temporary file for processing
            temp_file = self._create_temp_file(frame)
            if not temp_file:
                print("Failed to create temporary file")
                return None
            
            try:
                # Run inference
                results, orig_image = inference_model.predict_single(temp_file)
                results["original_image"] = orig_image
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                shutil.copy2(temp_file, SAVE_DIR / f"raw_{ts}.png")
                # Add quality metrics to results
                if quality:
                    results['frame_quality'] = quality
                    results['capture_timestamp'] = timestamp
                
                print(f"Signal processing completed - Quality: {results['overall_quality']}, "
                      f"Frame quality score: {quality['overall_score']:.2f}")
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
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='trigger_frame_')
            os.close(temp_fd)
            
            # Save frame to temporary file
            success = cv2.imwrite(temp_path.replace(".jpg", ".png"), frame)
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
        for temp_path in self.temp_files[:]:
            self._cleanup_temp_file(temp_path)
    
    def get_system_info(self):
        """Get information about current system state"""
        with self.buffer_lock:
            pre_trigger_size = len(self.pre_trigger_buffer)
            
        with self.post_trigger_lock:
            post_trigger_size = len(self.post_trigger_frames)
            
        return {
            'pre_trigger_size': pre_trigger_size,
            'post_trigger_size': post_trigger_size,
            'is_recording': self.is_recording,
            'is_post_trigger_recording': self.is_post_trigger_recording,
            'signal_received': self.signal_received,
            'temp_files_count': len(self.temp_files)
        }
    
    def reset_signal_state(self):
        """Reset signal state for next capture"""
        with self.processing_lock:
            self.signal_received = False
            self.signal_timestamp = None
            
        # Clear post-trigger data
        with self.post_trigger_lock:
            self.post_trigger_frames.clear()
            self.post_trigger_timestamps.clear()
            
        print("Signal state reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_recording()
        self.cleanup_all_temp_files() 
