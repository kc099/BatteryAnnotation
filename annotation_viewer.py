#!/usr/bin/env python3
"""
Battery Annotation Viewer with Tkinter GUI

This application allows users to upload an image and view the corresponding
masked annotations from the enhanced annotation JSON file.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class BatteryAnnotationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Annotation Viewer")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_image_path = None
        self.current_annotation_path = None
        self.original_image = None
        self.annotation_data = None
        
        # Colors for different classes
        self.class_colors = {
            1: (255, 0, 0),    # Plus knob - Red
            2: (0, 255, 0),    # Minus knob - Green
            3: (0, 0, 255),    # Text - Blue
            4: (255, 255, 0)   # Hole - Yellow
        }
        
        self.class_names = {
            1: "Plus Knob",
            2: "Minus Knob", 
            3: "Text",
            4: "Hole"
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Battery Annotation Viewer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Upload button
        upload_btn = ttk.Button(main_frame, text="Upload Image", 
                               command=self.upload_image)
        upload_btn.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # File path display
        self.file_path_var = tk.StringVar(value="No image selected")
        path_label = ttk.Label(main_frame, textvariable=self.file_path_var, 
                              wraplength=400)
        path_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # Show masks checkbox
        self.show_masks_var = tk.BooleanVar(value=True)
        masks_check = ttk.Checkbutton(control_frame, text="Show Masks", 
                                     variable=self.show_masks_var,
                                     command=self.update_display)
        masks_check.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Show boxes checkbox
        self.show_boxes_var = tk.BooleanVar(value=True)
        boxes_check = ttk.Checkbutton(control_frame, text="Show Bounding Boxes", 
                                     variable=self.show_boxes_var,
                                     command=self.update_display)
        boxes_check.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Show labels checkbox
        self.show_labels_var = tk.BooleanVar(value=True)
        labels_check = ttk.Checkbutton(control_frame, text="Show Labels", 
                                      variable=self.show_labels_var,
                                      command=self.update_display)
        labels_check.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Class visibility checkboxes
        class_frame = ttk.LabelFrame(control_frame, text="Class Visibility", padding="5")
        class_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.class_visibility = {}
        for i, (class_id, class_name) in enumerate(self.class_names.items()):
            var = tk.BooleanVar(value=True)
            self.class_visibility[class_id] = var
            check = ttk.Checkbutton(class_frame, text=class_name, variable=var,
                                   command=self.update_display)
            check.grid(row=i, column=0, sticky=tk.W, pady=1)
        
        # Image display area
        display_frame = ttk.LabelFrame(main_frame, text="Annotation Display", padding="10")
        display_frame.grid(row=1, column=1, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure for image display
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def upload_image(self):
        """Upload an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.current_image_path = file_path
        self.file_path_var.set(f"Image: {os.path.basename(file_path)}")
        
        # Try to find corresponding annotation file
        self.find_annotation_file()
        
        # Load and display the image
        self.load_image()
        
    def find_annotation_file(self):
        """Find the corresponding annotation file"""
        if not self.current_image_path:
            return
            
        image_path = Path(self.current_image_path)
        base_name = image_path.stem
        
        # Look for annotation file in the same directory
        annotation_file = image_path.parent / f"{base_name}_enhanced_annotation.json"
        
        if annotation_file.exists():
            self.current_annotation_path = str(annotation_file)
            self.status_var.set(f"Found annotation: {annotation_file.name}")
            self.load_annotation()
        else:
            # Look in parent directories
            for parent_dir in image_path.parents:
                annotation_file = parent_dir / f"{base_name}_enhanced_annotation.json"
                if annotation_file.exists():
                    self.current_annotation_path = str(annotation_file)
                    self.status_var.set(f"Found annotation: {annotation_file.name}")
                    self.load_annotation()
                    return
            
            self.current_annotation_path = None
            self.status_var.set("No annotation file found")
            messagebox.showwarning("Warning", 
                                 f"No annotation file found for {base_name}")
    
    def load_image(self):
        """Load the image file"""
        try:
            # Load image with OpenCV
            image = cv2.imread(self.current_image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.original_image = image
            
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def load_annotation(self):
        """Load the annotation JSON file"""
        if not self.current_annotation_path:
            return
            
        try:
            with open(self.current_annotation_path, 'r') as f:
                self.annotation_data = json.load(f)
            
            # Count components
            components = []
            if self.annotation_data.get('plus_knob_polygon'):
                components.append("Plus Knob")
            if self.annotation_data.get('minus_knob_polygon'):
                components.append("Minus Knob")
            if self.annotation_data.get('text_polygon'):
                components.append("Text")
            if self.annotation_data.get('hole_polygons'):
                components.append(f"{len(self.annotation_data['hole_polygons'])} Hole(s)")
            
            self.status_var.set(f"Loaded annotation with: {', '.join(components)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotation: {str(e)}")
            self.status_var.set("Error loading annotation")
            self.annotation_data = None
    
    def update_display(self):
        """Update the image display with annotations"""
        if self.original_image is None:
            return
            
        # Clear the plot
        self.ax.clear()
        
        # Display the original image
        self.ax.imshow(self.original_image)
        
        if self.annotation_data and self.show_boxes_var.get():
            self.draw_annotations()
        
        # Remove axes
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Battery Annotation Viewer", fontsize=14, fontweight='bold')
        
        # Update canvas
        self.canvas.draw()
    
    def draw_annotations(self):
        """Draw annotations on the image"""
        if not self.annotation_data:
            return
            
        # Draw masks if enabled
        if self.show_masks_var.get():
            self.draw_masks()
        
        # Draw bounding boxes
        if self.show_boxes_var.get():
            self.draw_bounding_boxes()
    
    def get_mpl_color(self, class_id):
        rgb = self.class_colors.get(class_id, (128, 128, 128))
        return tuple([c / 255.0 for c in rgb])

    def draw_masks(self):
        """Draw mask overlays from polygon data"""
        if not self.annotation_data:
            return
            
        # Draw plus knob polygon
        if self.class_visibility.get(1, True) and 'plus_knob_polygon' in self.annotation_data:
            polygon = self.annotation_data['plus_knob_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                color = self.get_mpl_color(1)
                self.ax.fill(polygon_array[:, 0], polygon_array[:, 1], 
                           color=color, alpha=0.3, label="Plus Knob")
        
        # Draw minus knob polygon
        if self.class_visibility.get(2, True) and 'minus_knob_polygon' in self.annotation_data:
            polygon = self.annotation_data['minus_knob_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                color = self.get_mpl_color(2)
                self.ax.fill(polygon_array[:, 0], polygon_array[:, 1], 
                           color=color, alpha=0.3, label="Minus Knob")
        
        # Draw text polygon
        if self.class_visibility.get(3, True) and 'text_polygon' in self.annotation_data:
            polygon = self.annotation_data['text_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                color = self.get_mpl_color(3)
                self.ax.fill(polygon_array[:, 0], polygon_array[:, 1], 
                           color=color, alpha=0.3, label="Text")
        
        # Draw hole polygons
        if self.class_visibility.get(4, True) and 'hole_polygons' in self.annotation_data:
            holes = self.annotation_data['hole_polygons']
            if holes:
                for hole in holes:
                    if hole and len(hole) >= 3:
                        hole_array = np.array(hole)
                        color = self.get_mpl_color(4)
                        self.ax.fill(hole_array[:, 0], hole_array[:, 1], 
                                   color=color, alpha=0.3, label="Hole")
    
    def draw_bounding_boxes(self):
        """Draw bounding boxes and labels from polygon data"""
        if not self.annotation_data:
            return
            
        # Draw plus knob bounding box
        if self.class_visibility.get(1, True) and 'plus_knob_polygon' in self.annotation_data:
            polygon = self.annotation_data['plus_knob_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                x1, y1 = np.min(polygon_array, axis=0)
                x2, y2 = np.max(polygon_array, axis=0)
                width = x2 - x1
                height = y2 - y1
                
                color = self.get_mpl_color(1)
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none')
                self.ax.add_patch(rect)
                
                if self.show_labels_var.get():
                    self.ax.text(x1, y1-5, "Plus Knob", 
                               color=color, fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='white', alpha=0.8))
        
        # Draw minus knob bounding box
        if self.class_visibility.get(2, True) and 'minus_knob_polygon' in self.annotation_data:
            polygon = self.annotation_data['minus_knob_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                x1, y1 = np.min(polygon_array, axis=0)
                x2, y2 = np.max(polygon_array, axis=0)
                width = x2 - x1
                height = y2 - y1
                
                color = self.get_mpl_color(2)
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none')
                self.ax.add_patch(rect)
                
                if self.show_labels_var.get():
                    self.ax.text(x1, y1-5, "Minus Knob", 
                               color=color, fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='white', alpha=0.8))
        
        # Draw text bounding box
        if self.class_visibility.get(3, True) and 'text_polygon' in self.annotation_data:
            polygon = self.annotation_data['text_polygon']
            if polygon and len(polygon) >= 3:
                polygon_array = np.array(polygon)
                x1, y1 = np.min(polygon_array, axis=0)
                x2, y2 = np.max(polygon_array, axis=0)
                width = x2 - x1
                height = y2 - y1
                
                color = self.get_mpl_color(3)
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, 
                                       facecolor='none')
                self.ax.add_patch(rect)
                
                if self.show_labels_var.get():
                    self.ax.text(x1, y1-5, "Text", 
                               color=color, fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor='white', alpha=0.8))
        
        # Draw hole bounding boxes
        if self.class_visibility.get(4, True) and 'hole_polygons' in self.annotation_data:
            holes = self.annotation_data['hole_polygons']
            if holes:
                for i, hole in enumerate(holes):
                    if hole and len(hole) >= 3:
                        hole_array = np.array(hole)
                        x1, y1 = np.min(hole_array, axis=0)
                        x2, y2 = np.max(hole_array, axis=0)
                        width = x2 - x1
                        height = y2 - y1
                        
                        color = self.get_mpl_color(4)
                        rect = patches.Rectangle((x1, y1), width, height, 
                                               linewidth=2, edgecolor=color, 
                                               facecolor='none')
                        self.ax.add_patch(rect)
                        
                        if self.show_labels_var.get():
                            label = f"Hole {i+1}" if len(holes) > 1 else "Hole"
                            self.ax.text(x1, y1-5, label, 
                                       color=color, fontsize=10, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor='white', alpha=0.8))
    
    def show_annotation_info(self):
        """Show detailed annotation information"""
        if not self.annotation_data:
            messagebox.showinfo("Info", "No annotation data loaded")
            return
            
        info_window = tk.Toplevel(self.root)
        info_window.title("Annotation Information")
        info_window.geometry("600x400")
        
        # Create text widget
        text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(info_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Format and display annotation data
        info_text = "Annotation Information:\n\n"
        
        # Basic info
        info_text += f"Image: {os.path.basename(self.current_image_path)}\n"
        info_text += f"Annotation: {os.path.basename(self.current_annotation_path)}\n"
        info_text += f"Image Size: {self.annotation_data.get('image_width', 'N/A')} x {self.annotation_data.get('image_height', 'N/A')}\n\n"
        
        # Quality information
        info_text += "Quality Information:\n"
        quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'surface_quality', 'overall_quality']
        for field in quality_fields:
            value = self.annotation_data.get(field, 'N/A')
            info_text += f"  {field.replace('_', ' ').title()}: {value}\n"
        info_text += "\n"
        
        # Component information
        info_text += "Components:\n"
        
        # Plus knob
        if 'plus_knob_polygon' in self.annotation_data and self.annotation_data['plus_knob_polygon']:
            info_text += "  ✓ Plus Knob (Red)\n"
            if 'plus_knob_area' in self.annotation_data:
                info_text += f"    Area: {self.annotation_data['plus_knob_area']:.1f} pixels\n"
        else:
            info_text += "  ✗ Plus Knob\n"
        
        # Minus knob
        if 'minus_knob_polygon' in self.annotation_data and self.annotation_data['minus_knob_polygon']:
            info_text += "  ✓ Minus Knob (Green)\n"
            if 'minus_knob_area' in self.annotation_data:
                info_text += f"    Area: {self.annotation_data['minus_knob_area']:.1f} pixels\n"
        else:
            info_text += "  ✗ Minus Knob\n"
        
        # Text
        if 'text_polygon' in self.annotation_data and self.annotation_data['text_polygon']:
            info_text += "  ✓ Text Area (Blue)\n"
            info_text += f"    Color Present: {self.annotation_data.get('text_color_present', 'N/A')}\n"
            info_text += f"    Readable: {self.annotation_data.get('text_readable', 'N/A')}\n"
            if 'text_content' in self.annotation_data and self.annotation_data['text_content']:
                info_text += f"    Content: {self.annotation_data['text_content']}\n"
        else:
            info_text += "  ✗ Text Area\n"
        
        # Holes
        holes = self.annotation_data.get('hole_polygons', [])
        if holes:
            info_text += f"  ✓ Holes (Yellow) - {len(holes)} found\n"
            hole_qualities = self.annotation_data.get('hole_qualities', [])
            if hole_qualities:
                info_text += f"    Qualities: {', '.join(hole_qualities)}\n"
        else:
            info_text += "  ✗ Holes\n"
        
        info_text += "\n"
        
        # Additional information
        if 'knob_size_correct' in self.annotation_data:
            info_text += f"Knob Size Correct: {self.annotation_data['knob_size_correct']}\n"
        
        if 'confidence_score' in self.annotation_data:
            info_text += f"Confidence Score: {self.annotation_data['confidence_score']:.3f}\n"
        
        if 'defect_types' in self.annotation_data and self.annotation_data['defect_types']:
            info_text += f"Defect Types: {', '.join(self.annotation_data['defect_types'])}\n"
        
        if 'notes' in self.annotation_data and self.annotation_data['notes']:
            info_text += f"Notes: {self.annotation_data['notes']}\n"
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = BatteryAnnotationViewer(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Image", command=app.upload_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # View menu
    view_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Annotation Info", command=app.show_annotation_info)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", 
                         command=lambda: messagebox.showinfo("About", 
                           "Battery Annotation Viewer\n\n"
                           "A tool for viewing battery quality annotations with masks and bounding boxes.\n\n"
                           "Features:\n"
                           "- Upload images and view annotations\n"
                           "- Toggle mask overlays\n"
                           "- Toggle bounding boxes and labels\n"
                           "- Class-specific visibility controls"))
    
    root.mainloop()

if __name__ == "__main__":
    main() 