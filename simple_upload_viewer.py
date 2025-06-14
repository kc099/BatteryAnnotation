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

# Import our inference class
from inference import BatteryInference

class SimpleUploadViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Quality Upload Viewer")
        self.root.geometry("1400x900")
        
        # Model and data
        self.inference_model = None
        self.model_path = None
        self.current_image_path = None
        self.current_results = None
        
        self.setup_ui()
    
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
        title_label = ttk.Label(main_frame, text="Battery Quality Upload Viewer", 
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
    
    def on_model_loaded(self, model_name):
        """Called when model is successfully loaded"""
        self.model_label.config(text=f"Model: {model_name}", foreground="green")
        self.upload_button.config(state="normal")
        messagebox.showinfo("Success", "Model loaded successfully!")
    
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
        
        # Add quality assessment overlay
        overlay_height = 150
        overlay = np.zeros((overlay_height, vis_image.shape[1], 3), dtype=np.uint8)
        overlay.fill(50)  # Dark gray background
        
        # Quality text
        quality_color = (0, 255, 0) if results['overall_quality'] == 'GOOD' else (0, 0, 255)
        cv2.putText(overlay, f"Overall Quality: {results['overall_quality']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
        
        # Component analysis
        hole_color = (0, 255, 0) if results['hole_good'] else (0, 0, 255)
        text_color = (0, 255, 0) if results['text_color_good'] else (0, 0, 255)
        knob_color = (0, 255, 0) if results['knob_size_good'] else (0, 0, 255)
        
        cv2.putText(overlay, f"Hole: {'GOOD' if results['hole_good'] else 'BAD'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hole_color, 2)
        cv2.putText(overlay, f"Text: {'GOOD' if results['text_color_good'] else 'BAD'}", 
                   (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(overlay, f"Knob: {'GOOD' if results['knob_size_good'] else 'BAD'}", 
                   (290, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, knob_color, 2)
        
        # Detection count
        cv2.putText(overlay, f"Objects Detected: {len(detections['boxes'])}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Combine original image with overlay
        result_image = np.vstack([vis_image, overlay])
        
        return result_image
    
    def update_results_text(self, results):
        """Update the detailed results text"""
        analysis = results['analysis_details']
        
        results_text = f"""🔍 DETAILED ANALYSIS RESULTS
{'='*50}

📊 OVERALL ASSESSMENT:
   Quality: {results['overall_quality']}

🕳️ HOLE ANALYSIS:
   Status: {'GOOD' if results['hole_good'] else 'BAD'}
   {analysis['hole_analysis']['status']}

🔘 KNOB ANALYSIS:
   Status: {'GOOD' if results['knob_size_good'] else 'BAD'}
   {analysis['knob_analysis']['status']}

📝 TEXT ANALYSIS:
   Status: {'GOOD' if results['text_color_good'] else 'BAD'}
   {analysis['text_analysis']['status']}

📦 OBJECT DETECTIONS:
   Total Objects: {len(results['detections']['boxes'])}
"""
        
        # Add detection details
        class_names = {0: 'background', 1: 'plus_knob', 2: 'minus_knob', 3: 'text_area', 4: 'hole'}
        for i, (score, label) in enumerate(zip(results['detections']['scores'], results['detections']['labels'])):
            results_text += f"   {i+1}. {class_names[label]}: {score:.3f}\n"
        
        # Add perspective points
        results_text += f"\n📍 PERSPECTIVE POINTS:\n"
        for i, (x, y) in enumerate(results['perspective_points']):
            results_text += f"   Point {i+1}: ({x:.1f}, {y:.1f})\n"
        
        # Update text widget
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

def main():
    root = tk.Tk()
    app = SimpleUploadViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 