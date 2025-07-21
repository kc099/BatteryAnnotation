#!/usr/bin/env python3
"""
Battery Quality Simple Upload Viewer GUI

Upload single images and view predictions with side-by-side comparison.
"""

import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
import torch
import time
import os

# Import our inference class
from inference import BatteryInference
from camera_streams import create_ip_camera
from signal_handler import SignalHandler

# Import capture systems
from video_buffer_capture import VideoBufferCapture
from pre_post_trigger_capture import PrePostTriggerCapture

class SimpleUploadViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Lid Quality Analyzer")
        self.root.geometry("1400x900")
        
        # Model and data
        self.inference_model = None
        self.model_path = None
        self.current_image_path = None
        self.current_results = None
        
        # Streaming state
        self.ip_camera = None
        self.is_streaming = False
        self.streaming_thread = None
        self.camera_url = "http://192.168.100.50:8080/stream-hd"  # Default IP camera URL
        self.capture_interval = 5.0  # Capture frame every 5 seconds
        self.last_capture_time = 0
        
        # Auto-capture control
        self.auto_capture_enabled = False
        self.auto_capture_thread = None
        
        # Signal handling state
        self.signal_processing_lock = threading.Lock()  # Prevent conflicts between auto-capture and signal capture
        
        # Capture mode system
        self.capture_mode = "Image"  # Default: Image, VideoBuffer, PrePostTrigger
        self.video_buffer_system = None
        self.pre_post_trigger_system = None
        
        # Signal handler for 24V signal via RS485/USB
        self.signal_handler = SignalHandler(signal_callback=self.on_signal_received)
        self.signal_handler.start_detection()
        
        self.setup_ui()

        # ── AUTO-LOAD DEFAULT MODEL ──────────────────────────────────────────
        DEFAULT_MODEL_PATH = "/home/nvidia/Amaron/BatteryAnnotation/best_custom_maskrcnn.pth"
        self._auto_load_default_model(DEFAULT_MODEL_PATH)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Battery Lid Quality Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(2, weight=1)
        
        # Model selection
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=0, padx=(0, 10))
        
        self.model_label = ttk.Label(control_frame, text="No model loaded", 
                                    foreground="gray")
        self.model_label.grid(row=0, column=1, padx=(0, 10))
        
        # Image upload
        self.upload_button = ttk.Button(control_frame, text="Upload Image", 
                                       command=self.upload_image, state="disabled")
        self.upload_button.grid(row=0, column=2, padx=(10, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Streaming controls
        stream_frame = ttk.LabelFrame(control_frame, text="IP Camera Streaming", padding="5")
        stream_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        stream_frame.columnconfigure(1, weight=1)
        
        # Camera URL input
        ttk.Label(stream_frame, text="Camera URL:").grid(row=0, column=0, padx=(0, 5))
        self.camera_url_var = tk.StringVar(value=self.camera_url)
        self.camera_url_entry = ttk.Entry(stream_frame, textvariable=self.camera_url_var, width=40)
        self.camera_url_entry.grid(row=0, column=1, padx=(0, 10))
        
        # Capture mode selection
        ttk.Label(stream_frame, text="Capture Mode:").grid(row=0, column=2, padx=(0, 5))
        self.capture_mode_var = tk.StringVar(value=self.capture_mode)
        self.capture_mode_combo = ttk.Combobox(stream_frame, textvariable=self.capture_mode_var, 
                                              values=["Image", "VideoBuffer", "PrePostTrigger"], 
                                              state="readonly", width=12)
        self.capture_mode_combo.grid(row=0, column=3, padx=(0, 10))
        self.capture_mode_combo.bind("<<ComboboxSelected>>", self.on_capture_mode_changed)
        
        # Capture interval input
        ttk.Label(stream_frame, text="Interval (sec):").grid(row=0, column=4, padx=(0, 5))
        self.capture_interval_var = tk.DoubleVar(value=self.capture_interval)
        self.capture_interval_entry = ttk.Entry(stream_frame, textvariable=self.capture_interval_var, width=8)
        self.capture_interval_entry.grid(row=0, column=5, padx=(0, 10))
        
        # Auto-capture toggle
        self.auto_capture_var = tk.BooleanVar(value=False)
        self.auto_capture_check = ttk.Checkbutton(stream_frame, text="Auto Capture", 
                                                 variable=self.auto_capture_var, 
                                                 command=self.toggle_auto_capture,
                                                 state=tk.DISABLED)
        self.auto_capture_check.grid(row=0, column=6, padx=(0, 10))
        
        # Streaming buttons
        self.start_stream_btn = ttk.Button(stream_frame, text="Start Streaming", 
                                         command=self.start_streaming, state=tk.DISABLED)
        self.start_stream_btn.grid(row=0, column=7, padx=(0, 5))
        
        self.stop_stream_btn = ttk.Button(stream_frame, text="Stop Streaming", 
                                        command=self.stop_streaming, state=tk.DISABLED)
        self.stop_stream_btn.grid(row=0, column=8, padx=(0, 5))
        
        self.capture_frame_btn = ttk.Button(stream_frame, text="Capture Frame", 
                                          command=self.capture_current_frame, state=tk.DISABLED)
        self.capture_frame_btn.grid(row=0, column=9, padx=(0, 5))
        
        # Streaming status
        self.stream_status_var = tk.StringVar(value="Streaming: Stopped")
        ttk.Label(stream_frame, textvariable=self.stream_status_var, 
                 font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=10, pady=(5, 0))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # Original image frame
        original_frame = ttk.LabelFrame(content_frame, text="Original Image", padding="10")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, bg='white', width=600, height=400)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction frame
        prediction_frame = ttk.LabelFrame(content_frame, text="Prediction Results", padding="10")
        prediction_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        prediction_frame.columnconfigure(0, weight=1)
        prediction_frame.rowconfigure(0, weight=1)
        
        # Create notebook for prediction tabs
        self.prediction_notebook = ttk.Notebook(prediction_frame)
        self.prediction_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Visualization tab
        viz_frame = ttk.Frame(self.prediction_notebook)
        self.prediction_notebook.add(viz_frame, text="Visualization")
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        self.prediction_canvas = tk.Canvas(viz_frame, bg='white', width=600, height=400)
        self.prediction_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results tab
        results_frame = ttk.Frame(self.prediction_notebook)
        self.prediction_notebook.add(results_frame, text="Detailed Results")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text widget
        self.results_text = tk.Text(results_frame, font=("Consolas", 10), wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initial placeholder text
        self.show_placeholder_content()
    
    def show_placeholder_content(self):
        """Show placeholder content when no image is loaded"""
        # Clear canvases
        self.original_canvas.delete("all")
        self.prediction_canvas.delete("all")
        
        # Add placeholder text
        self.original_canvas.create_text(300, 200, text="Upload an image to start", 
                                        font=("Arial", 14), fill="gray")
        self.prediction_canvas.create_text(300, 200, text="Predictions will appear here", 
                                          font=("Arial", 14), fill="gray")
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Upload an image and load a model to see detailed results...")
    
    def load_model(self):
        """Load the trained model"""
        model_path = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if not model_path:
            return
        
        def load_model_thread():
            try:
                self.progress.start()
                self.inference_model = BatteryInference(model_path)
                self.model_path = model_path
                
                # Update UI on main thread
                self.root.after(0, self.on_model_loaded, Path(model_path).name)
                
            except Exception as e:
                self.root.after(0, self.on_model_load_error, str(e))
            finally:
                self.root.after(0, self.progress.stop)
        
        # Load model in background thread
        threading.Thread(target=load_model_thread, daemon=True).start()
        
    def _auto_load_default_model(self, model_path):
        """Attempt to load a default model silently at startup."""
        if not os.path.isfile(model_path):
            print(f"[Auto-load] Model not found: {model_path}")
            return

        def _loader():
            try:
                self.progress.start()
                self.inference_model = BatteryInference(model_path)
                self.model_path = model_path
                self.root.after(0, self.on_model_loaded, Path(model_path).name)
            except Exception as e:
                print(f"[Auto-load] Failed: {e}")
            finally:
                self.root.after(0, self.progress.stop)

        threading.Thread(target=_loader, daemon=True).start()

    def on_model_loaded(self, model_name):
        """Called when model is successfully loaded"""
        self.model_label.config(text=f"Model: {model_name}", foreground="green")
        self.upload_button.config(state="normal")
        self.start_stream_btn.config(state="normal")  # Enable streaming when model is loaded
        self.auto_capture_check.config(state="normal")  # Enable auto-capture when model is loaded
        messagebox.showinfo("Success", "Model loaded successfully!")
    
    def on_capture_mode_changed(self, event=None):
        """Handle capture mode selection change"""
        new_mode = self.capture_mode_var.get()
        if new_mode != self.capture_mode:
            # Stop current systems if running
            if self.is_streaming:
                messagebox.showwarning("Warning", f"Please stop streaming before changing capture mode from {self.capture_mode} to {new_mode}")
                # Reset to previous mode
                self.capture_mode_var.set(self.capture_mode)
                return
            
            # Cleanup previous systems
            self._cleanup_capture_systems()
            
            # Update mode
            self.capture_mode = new_mode
            print(f"Capture mode changed to: {self.capture_mode}")
            
            # Initialize new system if needed
            if self.capture_mode in ["VideoBuffer", "PrePostTrigger"]:
                self._initialize_capture_system()
    
    def _initialize_capture_system(self):
        """Initialize the selected capture system"""
        try:
            if self.capture_mode == "VideoBuffer":
                self.video_buffer_system = VideoBufferCapture(buffer_duration=5.0, frame_rate=30)
                print("Video Buffer capture system initialized")
            elif self.capture_mode == "PrePostTrigger":
                self.pre_post_trigger_system = PrePostTriggerCapture(pre_trigger_duration=3.0, post_trigger_duration=2.0)
                print("Pre-Post Trigger capture system initialized")
        except Exception as e:
            print(f"Error initializing capture system: {e}")
    
    def _cleanup_capture_systems(self):
        """Cleanup capture systems"""
        try:
            if self.video_buffer_system:
                self.video_buffer_system.stop_recording()
                self.video_buffer_system.cleanup_all_temp_files()
                self.video_buffer_system = None
                print("Video Buffer system cleaned up")
                
            if self.pre_post_trigger_system:
                self.pre_post_trigger_system.stop_recording()
                self.pre_post_trigger_system.cleanup_all_temp_files()
                self.pre_post_trigger_system = None
                print("Pre-Post Trigger system cleaned up")
        except Exception as e:
            print(f"Error cleaning up capture systems: {e}")
    
    def toggle_auto_capture(self):
        """Toggle auto-capture on/off"""
        if not self.is_streaming:
            messagebox.showwarning("Warning", "Please start streaming first before enabling auto-capture")
            self.auto_capture_var.set(False)
            return
        
        self.auto_capture_enabled = self.auto_capture_var.get()
        
        if self.auto_capture_enabled:
            # Start auto-capture thread
            self.auto_capture_thread = threading.Thread(target=self._auto_capture_worker, daemon=True)
            self.auto_capture_thread.start()
            self.stream_status_var.set("Streaming: Active (Auto-capture ON)")
        else:
            # Stop auto-capture thread
            self.auto_capture_thread = None
            self.stream_status_var.set("Streaming: Active (Auto-capture OFF)")
    
    def on_model_load_error(self, error_msg):
        """Called when model loading fails"""
        messagebox.showerror("Error", f"Failed to load model:\n{error_msg}")
    
    def upload_image(self):
        """Upload and process an image"""
        if not self.inference_model:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if not image_path:
            return
        
        self.current_image_path = image_path
        
        def process_image_thread():
            try:
                self.progress.start()
                
                # Run inference
                results, orig_image = self.inference_model.predict_single(image_path)
                
                # Update UI on main thread
                self.root.after(0, self.on_prediction_complete, results, orig_image)
                
            except Exception as e:
                self.root.after(0, self.on_prediction_error, str(e))
            finally:
                self.root.after(0, self.progress.stop)
        
        # Process image in background thread
        threading.Thread(target=process_image_thread, daemon=True).start()
    
    def on_prediction_complete(self, results, orig_image):
        """Called when prediction is complete"""
        self.current_results = results
        
        # Display original image
        self.display_image(self.original_canvas, orig_image)
        
        # Create and display prediction visualization
        pred_image = self.create_prediction_visualization(orig_image, results)
        self.display_image(self.prediction_canvas, pred_image)
        
        # Update results text
        self.update_results_text(results)
        
        messagebox.showinfo("Success", "Image processed successfully!")
    
    def on_prediction_error(self, error_msg):
        """Called when prediction fails"""
        messagebox.showerror("Error", f"Failed to process image:\n{error_msg}")
    
    def display_image(self, canvas, image):
        """Display an image on a canvas with proper scaling"""
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 600, 400
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo
    
    def create_prediction_visualization(self, orig_image, results):
        """Create a visualization of the prediction results"""
        # Create a copy of the original image
        vis_image = orig_image.copy()
        
        # Draw bounding boxes
        detections = results['detections']
        class_names = {0: 'background', 1: 'plus_knob', 2: 'minus_knob', 3: 'text_area', 4: 'hole'}
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (box, score, label) in enumerate(zip(detections['boxes'], detections['scores'], detections['labels'])):
            x1, y1, x2, y2 = box.astype(int)
            color = colors[label % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{class_names[label]}: {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw perspective points
        persp = results['perspective_points']
        if np.any(persp > 0):
            for i, (x, y) in enumerate(persp.astype(int)):
                cv2.circle(vis_image, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(vis_image, f'P{i+1}', (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add quality assessment overlay - bigger and more visible
        overlay_height = 200
        overlay = np.zeros((overlay_height, vis_image.shape[1], 3), dtype=np.uint8)
        overlay.fill(40)  # Dark background for better contrast
        
        # Main quality result - LARGE and prominent
        quality_color = (0, 255, 0) if results['overall_quality'] == 'GOOD' else (0, 0, 255)
        quality_text = f"QUALITY: {results['overall_quality']}"
        
        # Calculate text size for centering
        (text_width, text_height), _ = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
        text_x = (vis_image.shape[1] - text_width) // 2
        
        # Add background rectangle for better visibility
        cv2.rectangle(overlay, (text_x - 20, 15), (text_x + text_width + 20, 15 + text_height + 20), 
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(overlay, (text_x - 20, 15), (text_x + text_width + 20, 15 + text_height + 20), 
                     quality_color, 3)  # Colored border
        
        # Main quality text - LARGE
        cv2.putText(overlay, quality_text, (text_x, 15 + text_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, quality_color, 4)
        
        # Component analysis - medium size
        hole_color = (0, 255, 0) if results['hole_good'] else (0, 0, 255)
        text_color = (0, 255, 0) if results['text_color_good'] else (0, 0, 255)
        knob_color = (0, 255, 0) if results['knob_size_good'] else (0, 0, 255)
        
        y_offset = 80
        cv2.putText(overlay, f"HOLE: {'GOOD' if results['hole_good'] else 'BAD'}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, hole_color, 3)
        cv2.putText(overlay, f"TEXT: {'GOOD' if results['text_color_good'] else 'BAD'}", 
                   (220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
        cv2.putText(overlay, f"KNOB: {'GOOD' if results['knob_size_good'] else 'BAD'}", 
                   (420, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, knob_color, 3)
        
        # Detection count and confidence
        y_offset = 120
        cv2.putText(overlay, f"Objects Detected: {len(detections['boxes'])}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show confidence scores
        if len(detections['scores']) > 0:
            avg_confidence = np.mean(detections['scores'])
            cv2.putText(overlay, f"Avg Confidence: {avg_confidence:.2f}", 
                       (300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add analysis details for text (if bad)
        if not results['text_color_good']:
            y_offset = 150
            analysis = results['analysis_details']['text_analysis']
            status_parts = analysis['status'].split(', ')
            detail_text = f"Text Issue: {status_parts[0] if status_parts else 'Unknown'}"
            cv2.putText(overlay, detail_text, 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Combine original image with overlay
        result_image = np.vstack([vis_image, overlay])
        
        return result_image
    
    def update_results_text(self, results):
        """Update the detailed results text"""
        analysis = results['analysis_details']
        
        results_text = f"""DETAILED ANALYSIS RESULTS
{'='*50}

OVERALL ASSESSMENT:
   Quality: {results['overall_quality']}

HOLE ANALYSIS:
   Status: {'GOOD' if results['hole_good'] else 'BAD'}
   {analysis['hole_analysis']['status']}

KNOB ANALYSIS:
   Status: {'GOOD' if results['knob_size_good'] else 'BAD'}
   {analysis['knob_analysis']['status']}

TEXT ANALYSIS:
   Status: {'GOOD' if results['text_color_good'] else 'BAD'}
   {analysis['text_analysis']['status']}

OBJECT DETECTIONS:
   Total Objects: {len(results['detections']['boxes'])}
"""
        
        # Add detection details
        class_names = {0: 'background', 1: 'plus_knob', 2: 'minus_knob', 3: 'text_area', 4: 'hole'}
        for i, (score, label) in enumerate(zip(results['detections']['scores'], results['detections']['labels'])):
            results_text += f"   {i+1}. {class_names[label]}: {score:.3f}\n"
        
        # Add perspective points
        results_text += f"\nPERSPECTIVE POINTS:\n"
        for i, (x, y) in enumerate(results['perspective_points']):
            results_text += f"   Point {i+1}: ({x:.1f}, {y:.1f})\n"
        
        # Update text widget
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
    
    def start_streaming(self):
        """Start IP camera streaming or buffer/trigger system"""
        if self.is_streaming:
            messagebox.showwarning("Warning", "Streaming is already active")
            return
        
        if not self.inference_model:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        # Get camera URL and interval from UI
        self.camera_url = self.camera_url_var.get()
        self.capture_interval = self.capture_interval_var.get()
        
        if not self.camera_url:
            messagebox.showerror("Error", "Please enter a camera URL")
            return
        
        try:
            # Initialize capture system based on mode
            if self.capture_mode in ["VideoBuffer", "PrePostTrigger"]:
                self._initialize_capture_system()
                
                # Start capture system recording
                if self.capture_mode == "VideoBuffer" and self.video_buffer_system:
                    success = self.video_buffer_system.start_recording(self.camera_url)
                    if not success:
                        raise Exception("Failed to start Video Buffer recording")
                elif self.capture_mode == "PrePostTrigger" and self.pre_post_trigger_system:
                    success = self.pre_post_trigger_system.start_recording(self.camera_url)
                    if not success:
                        raise Exception("Failed to start Pre-Post Trigger recording")
                # Disable capture frame button in buffer modes
                self.capture_frame_btn.config(state=tk.DISABLED)
            else:
                # Image mode - use original IP camera
                self.ip_camera = create_ip_camera(self.camera_url)
                self.ip_camera.start()
                # Start streaming thread (only for camera connection, not auto-capture)
                self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
                self.streaming_thread.start()
                self.capture_frame_btn.config(state=tk.NORMAL)
            
            self.is_streaming = True
            
            # Reset auto-capture state
            self.auto_capture_enabled = False
            self.auto_capture_var.set(False)
            
            # Update UI
            self.start_stream_btn.config(state=tk.DISABLED)
            self.stop_stream_btn.config(state=tk.NORMAL)
            self.stream_status_var.set(f"Streaming: Active ({self.capture_mode} mode)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start streaming: {str(e)}")
            self.is_streaming = False

    def stop_streaming(self):
        """Stop IP camera streaming or buffer/trigger system"""
        if not self.is_streaming:
            return
        
        try:
            # Stop auto-capture first
            self.auto_capture_enabled = False
            self.auto_capture_var.set(False)
            self.auto_capture_thread = None
            
            # Stop capture systems based on mode
            if self.capture_mode == "VideoBuffer" and self.video_buffer_system:
                self.video_buffer_system.stop_recording()
                self.video_buffer_system.cleanup_all_temp_files()
            elif self.capture_mode == "PrePostTrigger" and self.pre_post_trigger_system:
                self.pre_post_trigger_system.stop_recording()
                self.pre_post_trigger_system.cleanup_all_temp_files()
            else:
                # Image mode - stop IP camera
                if self.ip_camera:
                    self.ip_camera.stop()
                    self.ip_camera = None
            
            self.is_streaming = False
            
            # Update UI
            self.start_stream_btn.config(state=tk.NORMAL)
            self.stop_stream_btn.config(state=tk.DISABLED)
            self.capture_frame_btn.config(state=tk.DISABLED)
            self.stream_status_var.set("Streaming: Stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop streaming: {str(e)}")
    
    def _streaming_worker(self):
        """Worker thread for maintaining camera connection only"""
        while self.is_streaming and self.ip_camera:
            try:
                # Just keep the camera connection alive
                frame = self.ip_camera.get_frame()
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in streaming worker: {e}")
                time.sleep(1.0)  # Wait before retry
    
    def _auto_capture_worker(self):
        """Worker thread for auto-capture functionality"""
        self.last_capture_time = time.time()
        
        while self.auto_capture_enabled and self.is_streaming and self.ip_camera:
            try:
                current_time = time.time()
                if current_time - self.last_capture_time >= self.capture_interval:
                    # Acquire lock to prevent conflicts with signal capture
                    with self.signal_processing_lock:
                        frame = self.ip_camera.get_frame()
                        if frame is not None:
                            self._process_streamed_frame(frame, "auto_capture")
                            self.last_capture_time = current_time
                
                # Small delay to prevent CPU hogging
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in auto-capture worker: {e}")
                time.sleep(1.0)  # Wait before retry
    
    def _process_streamed_frame(self, frame, capture_type="manual"):
        """Process a frame from the IP camera stream (Image mode)"""
        import tempfile
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a temporary file for the frame
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix=f'{capture_type}_frame_')
            os.close(temp_fd)
            cv2.imwrite(temp_path, frame)
            
            # Run inference on the frame
            results, orig_image = self.inference_model.predict_single(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Update UI on main thread
            self.root.after(0, self.on_prediction_complete, results, orig_image)
            
            # Update status based on capture type
            if capture_type == "signal":
                status_text = f"Signal Capture"
            elif capture_type == "auto_capture":
                status_text = f"Auto Capture"
            else:
                status_text = f"Manual Capture"
            
            self.root.after(0, lambda: self.stream_status_var.set(status_text))
            
        except Exception as e:
            print(f"Error processing streamed frame: {e}")
    
    def capture_current_frame(self):
        """Manually capture the current frame from the stream"""
        if not self.is_streaming or not self.ip_camera:
            messagebox.showwarning("Warning", "No active stream to capture from")
            return
        
        try:
            # Get current frame from IP camera
            frame = self.ip_camera.get_frame()
            
            if frame is not None:
                # Process the frame immediately
                self._process_streamed_frame(frame, "manual")
            else:
                messagebox.showwarning("Warning", "No frame available from camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture frame: {str(e)}")
    
    def on_signal_received(self, signal_type=""):
        """Callback for 24V signal (MODBUS_FRAME)"""
        if signal_type == "MODBUS_FRAME":
            print("24V Signal received - Processing...")
            
            # Check if model is loaded
            if not self.inference_model:
                print("No model loaded - cannot process signal")
                return
            
            # Check if streaming is active, if not start it
            if not self.is_streaming:
                print("Streaming not active - starting streaming...")
                self.root.after(0, self._start_streaming_for_signal)
                return
            
            # Process the signal-triggered capture based on mode
            if self.capture_mode == "VideoBuffer":
                self._process_video_buffer_signal()
            elif self.capture_mode == "PrePostTrigger":
                self._process_pre_post_trigger_signal()
            else:
                # Image mode - use original method
                self._process_signal_capture()
    
    def _start_streaming_for_signal(self):
        """Start streaming when triggered by 24V signal"""
        try:
            # Get camera URL from UI
            self.camera_url = self.camera_url_var.get()
            
            if not self.camera_url:
                print("No camera URL configured")
                return
            
            # Initialize capture system based on mode
            if self.capture_mode in ["VideoBuffer", "PrePostTrigger"]:
                self._initialize_capture_system()
                
                # Start capture system recording
                if self.capture_mode == "VideoBuffer" and self.video_buffer_system:
                    success = self.video_buffer_system.start_recording(self.camera_url)
                    if not success:
                        raise Exception("Failed to start Video Buffer recording")
                elif self.capture_mode == "PrePostTrigger" and self.pre_post_trigger_system:
                    success = self.pre_post_trigger_system.start_recording(self.camera_url)
                    if not success:
                        raise Exception("Failed to start Pre-Post Trigger recording")
            else:
                # Image mode - use original IP camera
                self.ip_camera = create_ip_camera(self.camera_url)
                self.ip_camera.start()
                
                # Start streaming thread
                self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
                self.streaming_thread.start()
            
            self.is_streaming = True
            
            # Update UI
            self.start_stream_btn.config(state=tk.DISABLED)
            self.stop_stream_btn.config(state=tk.NORMAL)
            self.capture_frame_btn.config(state=tk.NORMAL)
            self.stream_status_var.set(f"Streaming: Active ({self.capture_mode} mode - Signal-triggered)")
            
            # Wait a moment for camera to initialize, then process signal
            if self.capture_mode == "VideoBuffer":
                self.root.after(2000, self._process_video_buffer_signal)
            elif self.capture_mode == "PrePostTrigger":
                self.root.after(2000, self._process_pre_post_trigger_signal)
            else:
                self.root.after(2000, self._process_signal_capture)
            
        except Exception as e:
            print(f"Failed to start streaming for signal: {e}")
    
    def _process_signal_capture(self):
        """Process the signal-triggered capture (Image mode)"""
        try:
            # Acquire lock to prevent conflicts with auto-capture
            with self.signal_processing_lock:
                frame = self.capture_image_from_camera()
                if frame is not None:
                    # Process the frame with signal capture type
                    self._process_streamed_frame(frame, "signal")
                    
                    # Get the results from the last processed frame
                    if hasattr(self, 'current_results') and self.current_results:
                        # Determine GOOD/BAD (1/0)
                        result_value = 1 if self.current_results['overall_quality'] == 'GOOD' else 0
                        # Send result via Modbus frame
                        self.send_modbus_result(result_value)
                        print(f"Signal processing complete - Result: {result_value} ({self.current_results['overall_quality']})")
                    else:
                        print("No results available for signal processing")
                else:
                    print("No frame available for signal capture")
                    
        except Exception as e:
            print(f"Error during signal-triggered capture: {e}")
    
    def _process_video_buffer_signal(self):
        """Process signal using Video Buffer system"""
        try:
            if not self.video_buffer_system:
                print("Video Buffer system not initialized")
                return
            
            # Notify video buffer system of signal
            self.video_buffer_system.on_signal_received()
            
            # Process the signal capture
            results = self.video_buffer_system.process_signal_capture(self.inference_model)
            
            if results:
                # Update UI with results (show captured frame in original canvas, processed in processed canvas)
                orig_img = results.get('original_image') or results.get('orig_image')
                self.root.after(0, self.on_prediction_complete, results, orig_img)
                
                # Send result via Modbus
                result_value = 1 if results['overall_quality'] == 'GOOD' else 0
                self.send_modbus_result(result_value)
                
                # Reset signal state for next capture
                self.video_buffer_system.reset_signal_state()
                
                print(f"Video Buffer signal processing complete - Result: {result_value} ({results['overall_quality']})")
            else:
                print("Video Buffer signal processing failed")
                
        except Exception as e:
            print(f"Error during Video Buffer signal processing: {e}")
    
    def _process_pre_post_trigger_signal(self):
        """Process signal using Pre-Post Trigger system"""
        try:
            if not self.pre_post_trigger_system:
                print("Pre-Post Trigger system not initialized")
                return

            # Notify pre-post trigger system of signal
            self.pre_post_trigger_system.on_signal_received()

            # Wait for post-trigger duration + small buffer, then process
            post_trigger_duration = self.pre_post_trigger_system.post_trigger_duration
            buffer_time = 0.5  # seconds
            total_wait = int((post_trigger_duration + buffer_time) * 1000)  # ms

            def process_after_wait():
                results = self.pre_post_trigger_system.process_signal_capture(self.inference_model)
                if results:
                    orig_img = results.get('original_image') or results.get('orig_image')
                    self.root.after(0, self.on_prediction_complete, results, orig_img)
                    result_value = 1 if results['overall_quality'] == 'GOOD' else 0
                    self.send_modbus_result(result_value)
                    self.pre_post_trigger_system.reset_signal_state()
                    print(f"Pre-Post Trigger signal processing complete - Result: {result_value} ({results['overall_quality']})")
                else:
                    print("Pre-Post Trigger signal processing failed")

            self.root.after(total_wait, process_after_wait)

        except Exception as e:
            print(f"Error during Pre-Post Trigger signal processing: {e}")

    def capture_image_from_camera(self):
        """Capture a frame from the IP camera (if streaming)"""
        if self.ip_camera:
            try:
                frame = self.ip_camera.get_frame()
                return frame
            except Exception as e:
                print(f"Error capturing frame from camera: {e}")
                return None
        return None

    def send_modbus_result(self, result_value):
        """Send a Modbus frame with the result (1=GOOD, 0=BAD) via RS485/USB"""
        if hasattr(self, 'signal_handler') and self.signal_handler:
            try:
                # Use the new simplified method
                success = self.signal_handler.send_battery_quality_result(result_value)
                if success:
                    print(f"Successfully sent battery quality result: {result_value} ({'GOOD' if result_value else 'BAD'})")
                else:
                    print("Failed to send battery quality result")
            except Exception as e:
                print(f"Error sending Modbus result: {e}")
        else:
            print("Signal handler not available for Modbus result sending.")

    def cleanup(self):
        """Cleanup resources when application closes"""
        # Stop auto-capture
        self.auto_capture_enabled = False
        self.auto_capture_thread = None
        
        # Stop streaming
        if self.is_streaming:
            self.stop_streaming()
        
        # Cleanup capture systems
        self._cleanup_capture_systems()
        
        # Stop signal handler
        if hasattr(self, 'signal_handler') and self.signal_handler:
            self.signal_handler.stop_detection()

def main():
    root = tk.Tk()
    app = SimpleUploadViewer(root)
    
    # Bind cleanup to window close event
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 