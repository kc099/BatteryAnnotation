import cv2
import numpy as np
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
import math
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point
import yaml

@dataclass
class AdvancedAnnotation:
    """Enhanced annotation structure with polygon support"""
    image_path: str
    image_width: int
    image_height: int
    
    # Holes as polygons - List of polygons, each polygon is list of (x,y) points
    hole_polygons: List[List[Tuple[int, int]]] = field(default_factory=list)
    hole_qualities: List[str] = field(default_factory=list)  # "good", "deformed", "blocked"
    
    # Text as oriented bounding box (x_center, y_center, width, height, angle)
    text_obb: Optional[Dict[str, float]] = None
    text_polygon: Optional[List[Tuple[int, int]]] = None  # Alternative polygon annotation
    text_color_present: bool = False
    text_readable: bool = False
    text_content: str = ""  # Actual text if readable
    
    # Knobs as polygons
    plus_knob_polygon: Optional[List[Tuple[int, int]]] = None
    minus_knob_polygon: Optional[List[Tuple[int, int]]] = None
    knob_size_ratio: Optional[float] = None
    knob_size_ratio_correct: bool = False
    
    # Perspective correction points (4 corners of the plate)
    perspective_points: Optional[List[Tuple[int, int]]] = None
    
    # Overall quality and detailed defects
    overall_quality: str = "UNKNOWN"  # GOOD, BAD, UNKNOWN
    defect_types: List[str] = field(default_factory=list)  # List of specific defects
    confidence_score: float = 1.0  # Annotator confidence
    notes: str = ""

class AdvancedBatteryLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Battery Cover Quality Inspection Tool")
        self.root.geometry("1600x900")
        
        # State variables
        self.current_image = None
        self.current_image_path = None
        self.current_annotation = None
        self.image_list = []
        self.current_index = 0
        self.scale_factor = 1.0
        
        # Annotation state
        self.annotation_mode = "hole_polygon"
        self.current_polygon = []
        self.drawing = False
        self.selected_item = None
        
        # Defect types
        self.defect_types = [
            "missing_hole", "deformed_hole", "blocked_hole",
            "missing_text", "faded_text", "wrong_text_color",
            "wrong_knob_size", "damaged_knob", "missing_knob",
            "surface_defect", "color_mismatch", "warped_shape"
        ]
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create enhanced user interface"""
        # Main container with notebook for tabs
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Image display
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, width=900, height=700, bg="gray")
        self.canvas.pack(side=tk.LEFT)
        
        # Canvas scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=v_scrollbar.set)
        
        # Canvas bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<KeyPress-Delete>", self.delete_selected)
        self.canvas.bind("<Escape>", self.cancel_current_polygon)
        
        # Right panel - Controls in notebook
        control_notebook = ttk.Notebook(main_frame)
        control_notebook.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Tab 1: Basic Controls
        basic_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(basic_frame, text="Basic Controls")
        
        # File controls
        ttk.Label(basic_frame, text="File Controls", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Button(basic_frame, text="Load Images", command=self.load_images).grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(basic_frame, text="Save Annotations", command=self.save_annotations).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Navigation
        nav_frame = ttk.Frame(basic_frame)
        nav_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).grid(row=0, column=0, padx=5)
        self.image_label = ttk.Label(nav_frame, text="0/0")
        self.image_label.grid(row=0, column=1, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).grid(row=0, column=2, padx=5)
        
        # Annotation mode
        ttk.Label(basic_frame, text="Annotation Mode", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, pady=10)
        self.mode_var = tk.StringVar(value="hole_polygon")
        modes = [
            ("Hole Polygon", "hole_polygon"),
            ("Text Region", "text_region"),
            ("Plus Knob", "plus_knob"),
            ("Minus Knob", "minus_knob"),
            ("Perspective Points", "perspective")
        ]
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(basic_frame, text=text, variable=self.mode_var, 
                          value=value, command=self.change_mode).grid(row=5+i, column=0, sticky=tk.W, padx=20)
        
        # Tab 2: Quality Assessment
        quality_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(quality_frame, text="Quality Assessment")
        
        # Text quality
        ttk.Label(quality_frame, text="Text Quality", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        self.text_color_var = tk.BooleanVar()
        ttk.Checkbutton(quality_frame, text="White Text Present", 
                       variable=self.text_color_var).grid(row=1, column=0, sticky=tk.W, padx=20)
        self.text_readable_var = tk.BooleanVar()
        ttk.Checkbutton(quality_frame, text="Text Readable", 
                       variable=self.text_readable_var).grid(row=2, column=0, sticky=tk.W, padx=20)
        
        ttk.Label(quality_frame, text="Text Content:").grid(row=3, column=0, sticky=tk.W, padx=20)
        self.text_content_entry = ttk.Entry(quality_frame, width=20)
        self.text_content_entry.grid(row=4, column=0, padx=20)
        
        # Hole quality
        ttk.Label(quality_frame, text="Hole Quality", font=("Arial", 11, "bold")).grid(row=5, column=0, columnspan=2, pady=5)
        self.hole_list = tk.Listbox(quality_frame, height=6, width=30)
        self.hole_list.grid(row=6, column=0, columnspan=2, padx=20)
        self.hole_list.bind('<<ListboxSelect>>', self.on_hole_select)
        
        hole_quality_frame = ttk.Frame(quality_frame)
        hole_quality_frame.grid(row=7, column=0, columnspan=2)
        ttk.Label(hole_quality_frame, text="Selected hole:").grid(row=0, column=0)
        self.hole_quality_var = tk.StringVar(value="good")
        for i, quality in enumerate(["good", "deformed", "blocked"]):
            ttk.Radiobutton(hole_quality_frame, text=quality, variable=self.hole_quality_var,
                          value=quality, command=self.update_hole_quality).grid(row=0, column=i+1)
        
        # Knob analysis
        ttk.Label(quality_frame, text="Knob Analysis", font=("Arial", 11, "bold")).grid(row=8, column=0, columnspan=2, pady=5)
        self.knob_ratio_label = ttk.Label(quality_frame, text="Ratio: Not calculated")
        self.knob_ratio_label.grid(row=9, column=0, columnspan=2)
        ttk.Button(quality_frame, text="Calculate Knob Ratio", 
                  command=self.calculate_knob_ratio).grid(row=10, column=0, columnspan=2, pady=5)
        
        # Overall quality
        ttk.Label(quality_frame, text="Overall Quality", font=("Arial", 11, "bold")).grid(row=11, column=0, columnspan=2, pady=5)
        self.quality_var = tk.StringVar(value="UNKNOWN")
        quality_radio_frame = ttk.Frame(quality_frame)
        quality_radio_frame.grid(row=12, column=0, columnspan=2)
        ttk.Radiobutton(quality_radio_frame, text="Good", variable=self.quality_var, 
                       value="GOOD").grid(row=0, column=0)
        ttk.Radiobutton(quality_radio_frame, text="Bad", variable=self.quality_var, 
                       value="BAD").grid(row=0, column=1)
        
        # Confidence score
        ttk.Label(quality_frame, text="Annotation Confidence:").grid(row=13, column=0, pady=5)
        self.confidence_var = tk.DoubleVar(value=1.0)
        self.confidence_scale = ttk.Scale(quality_frame, from_=0.0, to=1.0, 
                                        variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.grid(row=14, column=0, columnspan=2, padx=20)
        
        # Tab 3: Defects
        defects_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(defects_frame, text="Defects")
        
        ttk.Label(defects_frame, text="Detected Defects", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=5)
        
        # Defect checkboxes
        self.defect_vars = {}
        for i, defect in enumerate(self.defect_types):
            var = tk.BooleanVar()
            self.defect_vars[defect] = var
            ttk.Checkbutton(defects_frame, text=defect.replace('_', ' ').title(), 
                          variable=var).grid(row=i+1, column=0, sticky=tk.W, padx=20)
        
        # Notes
        ttk.Label(defects_frame, text="Notes:").grid(row=len(self.defect_types)+2, column=0, pady=5)
        self.notes_text = tk.Text(defects_frame, height=4, width=30)
        self.notes_text.grid(row=len(self.defect_types)+3, column=0, padx=20)
        
        # Tab 4: Advanced
        advanced_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(advanced_frame, text="Advanced")
        
        # Perspective correction
        ttk.Label(advanced_frame, text="Perspective Correction", font=("Arial", 11, "bold")).grid(row=0, column=0, pady=5)
        ttk.Button(advanced_frame, text="Apply Perspective Correction", 
                  command=self.apply_perspective_correction).grid(row=1, column=0, pady=5)
        ttk.Button(advanced_frame, text="Auto-detect Plate Corners", 
                  command=self.auto_detect_corners).grid(row=2, column=0, pady=5)
        
        # Analysis tools
        ttk.Label(advanced_frame, text="Analysis Tools", font=("Arial", 11, "bold")).grid(row=3, column=0, pady=10)
        ttk.Button(advanced_frame, text="Analyze Hole Circularity", 
                  command=self.analyze_hole_circularity).grid(row=4, column=0, pady=5)
        ttk.Button(advanced_frame, text="Measure Distances", 
                  command=self.measure_distances).grid(row=5, column=0, pady=5)
        ttk.Button(advanced_frame, text="Color Analysis", 
                  command=self.analyze_colors).grid(row=6, column=0, pady=5)
        
        # Export options
        ttk.Label(advanced_frame, text="Export Options", font=("Arial", 11, "bold")).grid(row=7, column=0, pady=10)
        ttk.Button(advanced_frame, text="Export for Training", 
                  command=self.export_training_data).grid(row=8, column=0, pady=5)
        ttk.Button(advanced_frame, text="Generate Report", 
                  command=self.generate_report).grid(row=9, column=0, pady=5)
        
        # Action buttons at bottom
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=1, column=1, pady=10)
        ttk.Button(action_frame, text="Clear Current", command=self.clear_current).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="Auto Quality Check", command=self.auto_quality_check).grid(row=0, column=1, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Instructions
        self.show_instructions()
    
    def show_instructions(self):
        """Show usage instructions"""
        instructions = """
        Keyboard Shortcuts:
        - Click: Add polygon point
        - Double-click: Close polygon
        - Right-click: Undo last point
        - ESC: Cancel current polygon
        - Delete: Delete selected item
        
        Tips:
        - Use polygon mode for precise hole shapes
        - Mark 4 perspective points for tilt correction
        - Auto-analyze features after annotation
        """
        messagebox.showinfo("Instructions", instructions)
    
    def change_mode(self):
        """Change annotation mode"""
        self.annotation_mode = self.mode_var.get()
        self.current_polygon = []
        self.drawing = False
        self.update_display()
        self.status_var.set(f"Mode: {self.annotation_mode}")
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if self.current_image is None:
            return
        
        # Convert to image coordinates
        x = int(event.x / self.scale_factor)
        y = int(event.y / self.scale_factor)
        
        if self.annotation_mode in ["hole_polygon", "text_region", "plus_knob", "minus_knob"]:
            self.current_polygon.append((x, y))
            self.drawing = True
            self.update_display()
            
        elif self.annotation_mode == "perspective":
            if len(self.current_polygon) < 4:
                self.current_polygon.append((x, y))
                self.update_display()
                if len(self.current_polygon) == 4:
                    self.current_annotation.perspective_points = self.current_polygon.copy()
                    self.current_polygon = []
                    self.status_var.set("Perspective points set")
    
    def on_double_click(self, event):
        """Close current polygon on double click"""
        if self.drawing and len(self.current_polygon) >= 3:
            self.close_current_polygon()
    
    def on_canvas_motion(self, event):
        """Show preview while drawing polygon"""
        if self.drawing and self.current_polygon:
            self.update_display()
            # Draw preview line to current mouse position
            if len(self.current_polygon) > 0:
                x = int(event.x / self.scale_factor)
                y = int(event.y / self.scale_factor)
                last_x, last_y = self.current_polygon[-1]
                
                # Scale back for display
                x1 = int(last_x * self.scale_factor)
                y1 = int(last_y * self.scale_factor)
                x2 = int(x * self.scale_factor)
                y2 = int(y * self.scale_factor)
                
                self.canvas.create_line(x1, y1, x2, y2, fill="yellow", 
                                      width=2, tags="preview", dash=(5, 5))
    
    def close_current_polygon(self):
        """Close and save current polygon"""
        if len(self.current_polygon) < 3:
            return
        
        if self.annotation_mode == "hole_polygon":
            self.current_annotation.hole_polygons.append(self.current_polygon.copy())
            self.current_annotation.hole_qualities.append("good")
            self.update_hole_list()
            
        elif self.annotation_mode == "text_region":
            self.current_annotation.text_polygon = self.current_polygon.copy()
            # Calculate oriented bounding box
            self.current_annotation.text_obb = self.polygon_to_obb(self.current_polygon)
            
        elif self.annotation_mode == "plus_knob":
            self.current_annotation.plus_knob_polygon = self.current_polygon.copy()
            
        elif self.annotation_mode == "minus_knob":
            self.current_annotation.minus_knob_polygon = self.current_polygon.copy()
        
        self.current_polygon = []
        self.drawing = False
        self.update_display()
    
    def polygon_to_obb(self, polygon):
        """Convert polygon to oriented bounding box"""
        points = np.array(polygon, dtype=np.float32)
        rect = cv2.minAreaRect(points)
        (cx, cy), (width, height), angle = rect
        
        return {
            "center_x": float(cx),
            "center_y": float(cy),
            "width": float(width),
            "height": float(height),
            "angle": float(angle)
        }
    
    def calculate_polygon_area(self, polygon):
        """Calculate area of polygon using shoelace formula"""
        if len(polygon) < 3:
            return 0
        poly = ShapelyPolygon(polygon)
        return poly.area
    
    def calculate_knob_ratio(self):
        """Calculate ratio between knob sizes"""
        if (self.current_annotation.plus_knob_polygon and 
            self.current_annotation.minus_knob_polygon):
            
            plus_area = self.calculate_polygon_area(self.current_annotation.plus_knob_polygon)
            minus_area = self.calculate_polygon_area(self.current_annotation.minus_knob_polygon)
            
            if minus_area > 0:
                # Use square root of area ratio for linear dimension ratio
                ratio = math.sqrt(plus_area / minus_area)
                self.current_annotation.knob_size_ratio = ratio
                
                # Check if ratio is in acceptable range
                self.current_annotation.knob_size_ratio_correct = 1.15 <= ratio <= 1.35
                
                self.knob_ratio_label.config(
                    text=f"Ratio: {ratio:.3f} ({'GOOD' if self.current_annotation.knob_size_ratio_correct else 'BAD'})"
                )
                
                # Auto-update quality if ratio is bad
                if not self.current_annotation.knob_size_ratio_correct:
                    self.quality_var.set("BAD")
                    if "wrong_knob_size" not in self.defect_vars or not self.defect_vars["wrong_knob_size"].get():
                        self.defect_vars["wrong_knob_size"].set(True)
            else:
                self.knob_ratio_label.config(text="Ratio: Error - minus area is 0")
        else:
            self.knob_ratio_label.config(text="Ratio: Both knobs must be annotated")
    
    def analyze_hole_circularity(self):
        """Analyze how circular each hole is"""
        if not self.current_annotation.hole_polygons:
            messagebox.showinfo("Info", "No holes annotated")
            return
        
        results = []
        for i, polygon in enumerate(self.current_annotation.hole_polygons):
            if len(polygon) < 3:
                continue
            
            # Calculate circularity: 4π × area / perimeter²
            poly = ShapelyPolygon(polygon)
            area = poly.area
            perimeter = poly.length
            
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter ** 2)
                results.append((i, circularity))
                
                # Update hole quality based on circularity
                if circularity < 0.8:  # Threshold for deformed
                    self.current_annotation.hole_qualities[i] = "deformed"
                    self.update_hole_list()
        
        # Show results
        result_text = "Hole Circularity Analysis:\n"
        result_text += "(1.0 = perfect circle)\n\n"
        for i, circ in results:
            quality = self.current_annotation.hole_qualities[i]
            result_text += f"Hole {i+1}: {circ:.3f} ({quality})\n"
        
        messagebox.showinfo("Circularity Analysis", result_text)
    
    def auto_detect_corners(self):
        """Automatically detect plate corners using edge detection"""
        if self.current_image is None:
            return
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (presumably the plate)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                # Get the 4 corners
                corners = approx.reshape(-1, 2)
                
                # Sort corners: top-left, top-right, bottom-right, bottom-left
                sorted_corners = self.sort_corners(corners)
                
                self.current_annotation.perspective_points = [(int(x), int(y)) for x, y in sorted_corners]
                self.update_display()
                self.status_var.set("Auto-detected plate corners")
            else:
                messagebox.showwarning("Warning", "Could not detect 4 corners")
    
    def sort_corners(self, corners):
        """Sort corners in standard order"""
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Rearrange to start from top-left
        if sorted_corners[0][0] > center[0]:  # First point is on right
            sorted_corners = sorted_corners[1:] + sorted_corners[:1]
        
        return sorted_corners
    
    def apply_perspective_correction(self):
        """Apply perspective correction to the image"""
        if not self.current_annotation.perspective_points or len(self.current_annotation.perspective_points) != 4:
            messagebox.showwarning("Warning", "Please mark 4 perspective points first")
            return
        
        # Source points
        src_points = np.array(self.current_annotation.perspective_points, dtype=np.float32)
        
        # Calculate destination points (rectangle)
        width = int(np.linalg.norm(src_points[1] - src_points[0]))
        height = int(np.linalg.norm(src_points[3] - src_points[0]))
        
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        corrected = cv2.warpPerspective(self.current_image, matrix, (width, height))
        
        # Show result
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.current_image)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(corrected)
        plt.title("Perspective Corrected")
        plt.show()
        
        # Ask if user wants to use corrected image
        if messagebox.askyesno("Perspective Correction", "Use corrected image for annotation?"):
            self.current_image = corrected
            self.update_display()
    
    def auto_quality_check(self):
        """Automatically check quality based on annotations"""
        defects = []
        
        # Check holes
        if len(self.current_annotation.hole_polygons) < 10:  # Expected minimum holes
            defects.append("missing_hole")
        
        deformed_holes = sum(1 for q in self.current_annotation.hole_qualities if q == "deformed")
        if deformed_holes > 0:
            defects.append("deformed_hole")
        
        blocked_holes = sum(1 for q in self.current_annotation.hole_qualities if q == "blocked")
        if blocked_holes > 0:
            defects.append("blocked_hole")
        
        # Check text
        if not self.current_annotation.text_polygon:
            defects.append("missing_text")
        elif not self.current_annotation.text_color_present:
            defects.append("wrong_text_color")
        elif not self.current_annotation.text_readable:
            defects.append("faded_text")
        
        # Check knobs
        if not self.current_annotation.plus_knob_polygon:
            defects.append("missing_knob")
        if not self.current_annotation.minus_knob_polygon:
            defects.append("missing_knob")
        elif self.current_annotation.knob_size_ratio and not self.current_annotation.knob_size_ratio_correct:
            defects.append("wrong_knob_size")
        
        # Update defect checkboxes
        for defect_type, var in self.defect_vars.items():
            var.set(defect_type in defects)
        
        # Set overall quality
        if defects:
            self.quality_var.set("BAD")
            self.current_annotation.overall_quality = "BAD"
        else:
            self.quality_var.set("GOOD")
            self.current_annotation.overall_quality = "GOOD"
        
        self.status_var.set(f"Auto quality check complete. Found {len(defects)} defect(s)")
    
    def update_hole_list(self):
        """Update the hole list display"""
        self.hole_list.delete(0, tk.END)
        for i, (polygon, quality) in enumerate(zip(self.current_annotation.hole_polygons, 
                                                   self.current_annotation.hole_qualities)):
            area = self.calculate_polygon_area(polygon)
            self.hole_list.insert(tk.END, f"Hole {i+1}: {quality} (area: {area:.0f})")
    
    def on_hole_select(self, event):
        """Handle hole selection from list"""
        selection = self.hole_list.curselection()
        if selection:
            index = selection[0]
            self.selected_item = ("hole", index)
            quality = self.current_annotation.hole_qualities[index]
            self.hole_quality_var.set(quality)
            self.update_display()
    
    def update_hole_quality(self):
        """Update quality of selected hole"""
        if self.selected_item and self.selected_item[0] == "hole":
            index = self.selected_item[1]
            self.current_annotation.hole_qualities[index] = self.hole_quality_var.get()
            self.update_hole_list()
            self.update_display()
    
    def analyze_colors(self):
        """Analyze color distribution in annotated regions"""
        if not self.current_annotation.text_polygon:
            messagebox.showinfo("Info", "Please annotate text region first")
            return
        
        # Create mask for text region
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        points = np.array(self.current_annotation.text_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # Extract colors in text region
        text_region = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(text_region, cv2.COLOR_RGB2HSV)
        
        # Check for white color (high value, low saturation)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_pixels = cv2.countNonZero(white_mask)
        total_pixels = cv2.countNonZero(mask)
        
        white_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Update annotation
        self.current_annotation.text_color_present = white_percentage > 50
        self.text_color_var.set(self.current_annotation.text_color_present)
        
        # Show results
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(text_region)
        plt.title("Text Region")
        plt.subplot(1, 3, 2)
        plt.imshow(white_mask, cmap='gray')
        plt.title(f"White Pixels ({white_percentage:.1f}%)")
        plt.subplot(1, 3, 3)
        # Show color histogram
        colors = text_region[mask > 0]
        plt.hist(colors.reshape(-1, 3), bins=50, alpha=0.7, label=['R', 'G', 'B'])
        plt.legend()
        plt.title("Color Distribution")
        plt.show()
    
    def measure_distances(self):
        """Tool to measure distances between features"""
        messagebox.showinfo("Measure Tool", 
                          "Click two points to measure distance. This feature can be extended to measure distances between holes, knobs, etc.")
    
    def update_display(self):
        """Update canvas display with all annotations"""
        if self.current_image is None:
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Create display image
        display_image = self.current_image.copy()
        h, w = display_image.shape[:2]
        
        # Draw all annotations
        overlay = display_image.copy()
        
        # Draw holes
        for i, (polygon, quality) in enumerate(zip(self.current_annotation.hole_polygons, 
                                                  self.current_annotation.hole_qualities)):
            points = np.array(polygon, dtype=np.int32)
            color = {
                "good": (0, 255, 0),
                "deformed": (255, 165, 0),
                "blocked": (255, 0, 0)
            }.get(quality, (128, 128, 128))
            
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(display_image, [points], True, color, 2)
            
            # Highlight selected hole
            if self.selected_item and self.selected_item[0] == "hole" and self.selected_item[1] == i:
                cv2.polylines(display_image, [points], True, (255, 255, 0), 4)
            
            # Add label
            center = points.mean(axis=0).astype(int)
            cv2.putText(display_image, f"H{i+1}", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw text region
        if self.current_annotation.text_polygon:
            points = np.array(self.current_annotation.text_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (0, 255, 255))
            cv2.polylines(display_image, [points], True, (0, 255, 255), 3)
            
            # Draw OBB if available
            if self.current_annotation.text_obb:
                obb = self.current_annotation.text_obb
                box = cv2.boxPoints(((obb['center_x'], obb['center_y']), 
                                   (obb['width'], obb['height']), obb['angle']))
                box = np.int0(box)
                cv2.drawContours(display_image, [box], 0, (0, 200, 200), 2)
        
        # Draw knobs
        if self.current_annotation.plus_knob_polygon:
            points = np.array(self.current_annotation.plus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (255, 0, 255))
            cv2.polylines(display_image, [points], True, (255, 0, 255), 3)
            center = points.mean(axis=0).astype(int)
            cv2.putText(display_image, "+", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if self.current_annotation.minus_knob_polygon:
            points = np.array(self.current_annotation.minus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (0, 255, 255))
            cv2.polylines(display_image, [points], True, (0, 255, 255), 3)
            center = points.mean(axis=0).astype(int)
            cv2.putText(display_image, "-", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw perspective points
        if self.current_annotation.perspective_points:
            for i, point in enumerate(self.current_annotation.perspective_points):
                cv2.circle(display_image, point, 8, (255, 128, 0), -1)
                cv2.putText(display_image, f"P{i+1}", (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            
            # Draw perspective box
            points = np.array(self.current_annotation.perspective_points, dtype=np.int32)
            cv2.polylines(display_image, [points], True, (255, 128, 0), 2)
        
        # Draw current polygon being drawn
        if self.current_polygon:
            points = np.array(self.current_polygon, dtype=np.int32)
            cv2.polylines(display_image, [points], False, (255, 255, 0), 2)
            for point in self.current_polygon:
                cv2.circle(display_image, point, 4, (255, 255, 0), -1)
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
        
        # Add quality indicator
        quality_color = {
            "GOOD": (0, 255, 0),
            "BAD": (255, 0, 0),
            "UNKNOWN": (128, 128, 128)
        }.get(self.current_annotation.overall_quality, (128, 128, 128))
        
        cv2.rectangle(display_image, (10, 10), (150, 40), quality_color, -1)
        cv2.putText(display_image, self.current_annotation.overall_quality, 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize for display
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        display_image = cv2.resize(display_image, (new_w, new_h))
        
        # Convert to PhotoImage and display
        image_pil = Image.fromarray(display_image)
        self.photo = ImageTk.PhotoImage(image_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def on_right_click(self, event):
        """Handle right click - undo last point"""
        if self.drawing and self.current_polygon:
            self.current_polygon.pop()
            self.update_display()
    
    def cancel_current_polygon(self, event=None):
        """Cancel current polygon drawing"""
        self.current_polygon = []
        self.drawing = False
        self.update_display()
        self.status_var.set("Polygon cancelled")
    
    def delete_selected(self, event=None):
        """Delete selected annotation"""
        if self.selected_item:
            if self.selected_item[0] == "hole":
                index = self.selected_item[1]
                del self.current_annotation.hole_polygons[index]
                del self.current_annotation.hole_qualities[index]
                self.update_hole_list()
                self.selected_item = None
                self.update_display()
    
    def clear_current(self):
        """Clear current annotations"""
        if messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            self.current_annotation = AdvancedAnnotation(
                image_path=self.current_image_path,
                image_width=self.current_image.shape[1],
                image_height=self.current_image.shape[0]
            )
            self.hole_list.delete(0, tk.END)
            self.update_display()
    
    def load_images(self):
        """Load images from directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            self.image_list = []
            for ext in image_extensions:
                self.image_list.extend(Path(directory).glob(f"*{ext}"))
                self.image_list.extend(Path(directory).glob(f"*{ext.upper()}"))
            
            self.image_list = sorted(list(set(self.image_list)))
            
            if self.image_list:
                self.current_index = 0
                self.load_current_image()
                messagebox.showinfo("Success", f"Loaded {len(self.image_list)} images")
    
    def load_current_image(self):
        """Load and display current image"""
        if not self.image_list:
            return
        
        self.current_image_path = str(self.image_list[self.current_index])
        image = cv2.imread(self.current_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image
        
        # Calculate scale
        h, w = image.shape[:2]
        canvas_w, canvas_h = 900, 700
        self.scale_factor = min(canvas_w/w, canvas_h/h, 1.0)
        
        # Load or create annotation
        self.load_annotation()
        
        # Update displays
        self.update_display()
        self.update_hole_list()
        self.image_label.config(text=f"{self.current_index + 1}/{len(self.image_list)}")
    
    def load_annotation(self):
        """Load existing annotation"""
        annotation_path = self.current_image_path.replace(
            Path(self.current_image_path).suffix, "_advanced_annotation.json")
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                self.current_annotation = AdvancedAnnotation(**data)
                
                # Update UI
                self.text_color_var.set(self.current_annotation.text_color_present)
                self.text_readable_var.set(self.current_annotation.text_readable)
                self.text_content_entry.delete(0, tk.END)
                self.text_content_entry.insert(0, self.current_annotation.text_content)
                self.quality_var.set(self.current_annotation.overall_quality)
                self.confidence_var.set(self.current_annotation.confidence_score)
                
                # Update defects
                for defect in self.current_annotation.defect_types:
                    if defect in self.defect_vars:
                        self.defect_vars[defect].set(True)
                
                self.notes_text.delete(1.0, tk.END)
                self.notes_text.insert(1.0, self.current_annotation.notes)
        else:
            # Create new
            h, w = self.current_image.shape[:2]
            self.current_annotation = AdvancedAnnotation(
                image_path=self.current_image_path,
                image_width=w,
                image_height=h
            )
    
    def save_annotations(self):
        """Save current annotations"""
        if not self.current_annotation:
            return
        
        # Update from UI
        self.current_annotation.text_color_present = self.text_color_var.get()
        self.current_annotation.text_readable = self.text_readable_var.get()
        self.current_annotation.text_content = self.text_content_entry.get()
        self.current_annotation.overall_quality = self.quality_var.get()
        self.current_annotation.confidence_score = self.confidence_var.get()
        
        # Update defects
        self.current_annotation.defect_types = [
            defect for defect, var in self.defect_vars.items() if var.get()
        ]
        
        self.current_annotation.notes = self.notes_text.get(1.0, tk.END).strip()
        
        # Save
        annotation_path = self.current_image_path.replace(
            Path(self.current_image_path).suffix, "_advanced_annotation.json")
        
        with open(annotation_path, 'w') as f:
            json.dump(asdict(self.current_annotation), f, indent=2)
        
        self.status_var.set("Annotations saved!")
    
    def previous_image(self):
        """Navigate to previous image"""
        if self.image_list and self.current_index > 0:
            self.save_annotations()
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Navigate to next image"""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.save_annotations()
            self.current_index += 1
            self.load_current_image()
    
    def generate_report(self):
        """Generate detailed inspection report"""
        report = f"Battery Cover Inspection Report\n"
        report += f"Image: {Path(self.current_image_path).name}\n"
        report += f"Quality: {self.current_annotation.overall_quality}\n\n"
        
        report += f"Holes: {len(self.current_annotation.hole_polygons)}\n"
        good_holes = sum(1 for q in self.current_annotation.hole_qualities if q == "good")
        deformed_holes = sum(1 for q in self.current_annotation.hole_qualities if q == "deformed")
        blocked_holes = sum(1 for q in self.current_annotation.hole_qualities if q == "blocked")
        report += f"  - Good: {good_holes}\n"
        report += f"  - Deformed: {deformed_holes}\n"
        report += f"  - Blocked: {blocked_holes}\n\n"
        
        report += f"Text Region: {'Present' if self.current_annotation.text_polygon else 'Missing'}\n"
        if self.current_annotation.text_polygon:
            report += f"  - White Color: {'Yes' if self.current_annotation.text_color_present else 'No'}\n"
            report += f"  - Readable: {'Yes' if self.current_annotation.text_readable else 'No'}\n"
            report += f"  - Content: {self.current_annotation.text_content}\n\n"
        
        report += f"Knobs:\n"
        report += f"  - Plus Knob: {'Present' if self.current_annotation.plus_knob_polygon else 'Missing'}\n"
        report += f"  - Minus Knob: {'Present' if self.current_annotation.minus_knob_polygon else 'Missing'}\n"
        if self.current_annotation.knob_size_ratio:
            report += f"  - Size Ratio: {self.current_annotation.knob_size_ratio:.3f}\n"
            report += f"  - Ratio Correct: {'Yes' if self.current_annotation.knob_size_ratio_correct else 'No'}\n\n"
        
        if self.current_annotation.defect_types:
            report += f"Defects Found:\n"
            for defect in self.current_annotation.defect_types:
                report += f"  - {defect.replace('_', ' ').title()}\n"
        
        report += f"\nConfidence Score: {self.current_annotation.confidence_score:.2f}\n"
        report += f"Notes: {self.current_annotation.notes}\n"
        
        messagebox.showinfo("Inspection Report", report)
        
        # Save report
        report_path = self.current_image_path.replace(
            Path(self.current_image_path).suffix, "_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
    
    def export_training_data(self):
        """Export annotations for training"""
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        # Create export structure
        os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "annotations"), exist_ok=True)
        
        # Export all annotated images
        all_annotations = []
        for img_path in self.image_list:
            ann_path = str(img_path).replace(
                Path(img_path).suffix, "_advanced_annotation.json")
            
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                    all_annotations.append(ann_data)
                
                # Copy image
                img_name = Path(img_path).name
                cv2.imwrite(
                    os.path.join(export_dir, "images", img_name),
                    cv2.imread(str(img_path))
                )
                
                # Create masks
                ann = AdvancedAnnotation(**ann_data)
                self.create_training_masks(ann, export_dir, img_name)
        
        # Save consolidated annotations
        with open(os.path.join(export_dir, "all_annotations.json"), 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        # Create dataset statistics
        stats = {
            "total_images": len(all_annotations),
            "good_samples": sum(1 for a in all_annotations if a['overall_quality'] == 'GOOD'),
            "bad_samples": sum(1 for a in all_annotations if a['overall_quality'] == 'BAD'),
            "defect_distribution": {}
        }
        
        for ann in all_annotations:
            for defect in ann.get('defect_types', []):
                stats['defect_distribution'][defect] = stats['defect_distribution'].get(defect, 0) + 1
        
        with open(os.path.join(export_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        messagebox.showinfo("Export Complete", 
                          f"Exported {len(all_annotations)} annotations to {export_dir}")
    
    def create_training_masks(self, annotation, export_dir, image_name):
        """Create segmentation masks for training"""
        h, w = annotation.image_height, annotation.image_width
        
        # Hole mask
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        for i, (polygon, quality) in enumerate(zip(annotation.hole_polygons, annotation.hole_qualities)):
            points = np.array(polygon, dtype=np.int32)
            # Different values for different qualities
            value = {"good": 1, "deformed": 2, "blocked": 3}.get(quality, 1)
            cv2.fillPoly(hole_mask, [points], value * 50)
        
        # Text mask
        text_mask = np.zeros((h, w), dtype=np.uint8)
        if annotation.text_polygon:
            points = np.array(annotation.text_polygon, dtype=np.int32)
            cv2.fillPoly(text_mask, [points], 255)
        
        # Knob mask
        knob_mask = np.zeros((h, w), dtype=np.uint8)
        if annotation.plus_knob_polygon:
            points = np.array(annotation.plus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(knob_mask, [points], 100)
        if annotation.minus_knob_polygon:
            points = np.array(annotation.minus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(knob_mask, [points], 200)
        
        # Save masks
        base_name = Path(image_name).stem
        cv2.imwrite(os.path.join(export_dir, "masks", f"{base_name}_holes.png"), hole_mask)
        cv2.imwrite(os.path.join(export_dir, "masks", f"{base_name}_text.png"), text_mask)
        cv2.imwrite(os.path.join(export_dir, "masks", f"{base_name}_knobs.png"), knob_mask)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBatteryLabelingTool(root)
    root.mainloop()