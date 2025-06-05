import cv2
import numpy as np
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
import math
from collections import deque
import yaml

@dataclass
class AnnotationState:
    """Represents a state that can be undone/redone"""
    action_type: str
    data: Any
    timestamp: float

@dataclass
class EnhancedAnnotation:
    """Enhanced annotation structure with separate quality assessments"""
    image_path: str
    image_width: int
    image_height: int
    
    # Holes as polygons
    hole_polygons: List[List[Tuple[int, int]]] = field(default_factory=list)
    hole_qualities: List[str] = field(default_factory=list)
    
    # Text annotation
    text_polygon: Optional[List[Tuple[int, int]]] = None
    text_obb: Optional[Dict[str, float]] = None
    text_color_present: bool = False
    text_readable: bool = False
    text_content: str = ""
    
    # Knobs
    plus_knob_polygon: Optional[List[Tuple[int, int]]] = None
    minus_knob_polygon: Optional[List[Tuple[int, int]]] = None
    plus_knob_area: Optional[float] = None
    minus_knob_area: Optional[float] = None
    knob_size_correct: bool = False  # Plus should be larger than minus
    
    # Perspective points
    perspective_points: Optional[List[Tuple[int, int]]] = None
    
    # Separate quality assessments
    hole_quality: str = "UNKNOWN"  # GOOD, BAD, UNKNOWN
    text_quality: str = "UNKNOWN"
    knob_quality: str = "UNKNOWN"
    surface_quality: str = "UNKNOWN"
    overall_quality: str = "UNKNOWN"
    
    # Detailed defects
    defect_types: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    notes: str = ""

class ZoomableCanvas(tk.Canvas):
    """Custom canvas with zoom and pan functionality"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.pan_start = None
        self.original_image = None
        
        # Bind mouse events - Fix mouse wheel detection
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-2>", self.on_pan_start)  # Middle mouse button
        self.bind("<B2-Motion>", self.on_pan_drag)
        self.bind("<ButtonRelease-2>", self.on_pan_end)
        
        # For Mac/Linux - Fix wheel events
        self.bind("<Button-4>", self.on_mousewheel)
        self.bind("<Button-5>", self.on_mousewheel)
        
        # Add Control+wheel for zoom
        self.bind("<Control-MouseWheel>", self.on_mousewheel)
        self.focus_set()  # Allow keyboard focus
    
    def on_mousewheel(self, event):
        """Handle mouse wheel zoom"""
        # Check if Control is pressed for zoom, otherwise scroll
        if event.state & 0x4:  # Control key pressed
            if event.delta > 0 or event.num == 4:
                self.zoom_in(event)
            else:
                self.zoom_out(event)
        else:
            # Regular scrolling
            if event.delta > 0 or event.num == 4:
                self.yview_scroll(-1, "units")
            else:
                self.yview_scroll(1, "units")
    
    def zoom_in(self, event):
        """Zoom in"""
        if self.zoom_level < self.max_zoom:
            self.zoom_level = min(self.zoom_level * 1.2, self.max_zoom)
            self.event_generate("<<ZoomChanged>>")
    
    def zoom_out(self, event):
        """Zoom out"""
        if self.zoom_level > self.min_zoom:
            self.zoom_level = max(self.zoom_level / 1.2, self.min_zoom)
            self.event_generate("<<ZoomChanged>>")
    
    def reset_zoom(self):
        """Reset zoom to 1.0"""
        self.zoom_level = 1.0
        self.event_generate("<<ZoomChanged>>")
    
    def on_pan_start(self, event):
        """Start panning"""
        self.pan_start = (event.x, event.y)
        self.config(cursor="fleur")
    
    def on_pan_drag(self, event):
        """Pan the canvas"""
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            
            self.xview_scroll(-dx, "units")
            self.yview_scroll(-dy, "units")
            
            self.pan_start = (event.x, event.y)
    
    def on_pan_end(self, event):
        """End panning"""
        self.pan_start = None
        self.config(cursor="")
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates"""
        # Get the actual canvas coordinates accounting for scroll
        canvas_x = self.canvasx(canvas_x)
        canvas_y = self.canvasy(canvas_y)
        
        # Convert to image coordinates by dividing by zoom level
        img_x = canvas_x / self.zoom_level
        img_y = canvas_y / self.zoom_level
        
        return int(img_x), int(img_y)
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates"""
        canvas_x = img_x * self.zoom_level
        canvas_y = img_y * self.zoom_level
        return canvas_x, canvas_y

class EnhancedBatteryLabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Battery Annotation Tool with Zoom")
        self.root.geometry("1600x900")
        
        # State management
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        
        # Image and annotation state
        self.current_image = None
        self.current_image_path = None
        self.current_annotation = None
        self.image_list = []
        self.current_index = 0
        
        # Drawing state
        self.annotation_mode = "hole_polygon"
        self.current_polygon = []
        self.drawing = False
        self.selected_item = None
        
        # UI setup
        self.setup_ui()
        
    def setup_ui(self):
        """Create the enhanced user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Zoomable canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create zoomable canvas
        self.canvas = ZoomableCanvas(canvas_frame, width=900, height=700, bg="gray")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Zoom controls
        zoom_frame = ttk.Frame(canvas_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window).pack(side=tk.LEFT, padx=5)
        
        # Canvas event bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<<ZoomChanged>>", self.on_zoom_changed)
        
        # Ensure canvas can receive focus for keyboard events
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set(), add="+")
        
        # Keyboard bindings
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Escape>", lambda e: self.cancel_current_polygon())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        
        # Right panel - Controls in notebook
        control_notebook = ttk.Notebook(main_frame)
        control_notebook.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5)
        
        # Tab 1: Basic Controls
        basic_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(basic_frame, text="Basic")
        
        # File controls
        ttk.Label(basic_frame, text="File Controls", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Button(basic_frame, text="Load Images", command=self.load_images).pack(pady=2)
        ttk.Button(basic_frame, text="Save Annotations", command=self.save_annotations).pack(pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(basic_frame)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text="◄ Previous", command=self.previous_image).grid(row=0, column=0, padx=5)
        self.image_label = ttk.Label(nav_frame, text="0/0")
        self.image_label.grid(row=0, column=1, padx=5)
        ttk.Button(nav_frame, text="Next ►", command=self.next_image).grid(row=0, column=2, padx=5)
        
        # Undo/Redo
        undo_frame = ttk.Frame(basic_frame)
        undo_frame.pack(pady=10)
        ttk.Button(undo_frame, text="↶ Undo (Ctrl+Z)", command=self.undo).grid(row=0, column=0, padx=5)
        ttk.Button(undo_frame, text="↷ Redo (Ctrl+Y)", command=self.redo).grid(row=0, column=1, padx=5)
        
        # Annotation mode
        ttk.Label(basic_frame, text="Annotation Mode", font=("Arial", 11, "bold")).pack(pady=10)
        self.mode_var = tk.StringVar(value="hole_polygon")
        modes = [
            ("🔵 Hole Polygon", "hole_polygon"),
            ("📝 Text Region", "text_region"),
            ("➕ Plus Knob", "plus_knob"),
            ("➖ Minus Knob", "minus_knob"),
            ("📐 Perspective Points", "perspective")
        ]
        for text, value in modes:
            ttk.Radiobutton(basic_frame, text=text, variable=self.mode_var, 
                          value=value, command=self.change_mode).pack(anchor=tk.W, padx=20)
        
        # Tab 2: Component Quality
        quality_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(quality_frame, text="Component Quality")
        
        # Create quality sections
        self.create_quality_section(quality_frame, "Holes", 0)
        self.create_quality_section(quality_frame, "Text", 4)
        self.create_quality_section(quality_frame, "Knobs", 8)
        self.create_quality_section(quality_frame, "Surface", 12)
        
        # Tab 3: Hole Analysis
        hole_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(hole_frame, text="Holes")
        
        ttk.Label(hole_frame, text="Hole List", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Hole list with scrollbar
        list_frame = ttk.Frame(hole_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.hole_list = tk.Listbox(list_frame, height=15, yscrollcommand=scrollbar.set)
        self.hole_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.hole_list.yview)
        
        self.hole_list.bind('<<ListboxSelect>>', self.on_hole_select)
        
        # Hole quality controls
        hole_quality_frame = ttk.Frame(hole_frame)
        hole_quality_frame.pack(pady=10)
        
        ttk.Label(hole_quality_frame, text="Selected hole quality:").pack()
        self.hole_quality_var = tk.StringVar(value="good")
        quality_buttons = ttk.Frame(hole_quality_frame)
        quality_buttons.pack()
        
        for quality in ["good", "deformed", "blocked"]:
            ttk.Radiobutton(quality_buttons, text=quality.title(), 
                          variable=self.hole_quality_var, value=quality,
                          command=self.update_hole_quality).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(hole_frame, text="Analyze Circularity", 
                  command=self.analyze_hole_circularity).pack(pady=10)
        
        # Tab 4: Knob Analysis
        knob_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(knob_frame, text="Knobs")
        
        ttk.Label(knob_frame, text="Knob Size Analysis", font=("Arial", 11, "bold")).pack(pady=5)
        
        # Knob info display
        self.knob_info_text = tk.Text(knob_frame, height=8, width=40)
        self.knob_info_text.pack(pady=10)
        
        ttk.Button(knob_frame, text="Calculate Knob Sizes", 
                  command=self.calculate_knob_sizes).pack(pady=5)
        
        # Knob quality
        ttk.Label(knob_frame, text="Is Plus terminal larger than Minus?").pack(pady=10)
        self.knob_correct_var = tk.BooleanVar()
        ttk.Checkbutton(knob_frame, text="Yes (Correct sizing)", 
                       variable=self.knob_correct_var,
                       command=self.update_knob_quality).pack()
        
        # Tab 5: Overall Assessment
        overall_frame = ttk.Frame(control_notebook, padding="10")
        control_notebook.add(overall_frame, text="Overall")
        
        ttk.Label(overall_frame, text="Overall Quality Assessment", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Overall quality based on components
        self.overall_var = tk.StringVar(value="UNKNOWN")
        overall_radio_frame = ttk.Frame(overall_frame)
        overall_radio_frame.pack(pady=10)
        
        for value in ["GOOD", "BAD", "UNKNOWN"]:
            ttk.Radiobutton(overall_radio_frame, text=value, 
                          variable=self.overall_var, value=value).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(overall_frame, text="Auto-Calculate Overall Quality", 
                  command=self.calculate_overall_quality).pack(pady=10)
        
        # Confidence
        ttk.Label(overall_frame, text="Annotation Confidence:").pack(pady=5)
        self.confidence_var = tk.DoubleVar(value=1.0)
        self.confidence_scale = ttk.Scale(overall_frame, from_=0.0, to=1.0, 
                                        variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.pack(fill=tk.X, padx=20)
        self.confidence_label = ttk.Label(overall_frame, text="100%")
        self.confidence_label.pack()
        
        self.confidence_scale.config(command=lambda v: self.confidence_label.config(
            text=f"{float(v)*100:.0f}%"))
        
        # Notes
        ttk.Label(overall_frame, text="Notes:").pack(pady=5)
        self.notes_text = tk.Text(overall_frame, height=6, width=40)
        self.notes_text.pack(padx=20)
        
        # Actions
        action_frame = ttk.Frame(overall_frame)
        action_frame.pack(pady=20)
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export", command=self.export_training_data).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Show initial help
        self.show_help()
    
    def create_quality_section(self, parent, name, start_row):
        """Create a quality assessment section"""
        ttk.Label(parent, text=f"{name} Quality", font=("Arial", 10, "bold")).grid(
            row=start_row, column=0, columnspan=3, pady=10)
        
        quality_var = tk.StringVar(value="UNKNOWN")
        setattr(self, f"{name.lower()}_quality_var", quality_var)
        
        frame = ttk.Frame(parent)
        frame.grid(row=start_row+1, column=0, columnspan=3)
        
        for i, value in enumerate(["GOOD", "BAD", "UNKNOWN"]):
            ttk.Radiobutton(frame, text=value, variable=quality_var, 
                          value=value, command=lambda n=name: self.update_component_quality(n)).grid(
                          row=0, column=i, padx=5)
        
        # Add separator
        ttk.Separator(parent, orient='horizontal').grid(
            row=start_row+2, column=0, columnspan=3, sticky='ew', pady=10)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
Enhanced Battery Annotation Tool - Quick Guide

🖱️ Mouse Controls:
• Left Click: Add polygon point
• Double Click: Close polygon
• Right Click: Remove last point
• Ctrl + Mouse Wheel: Zoom in/out
• Mouse Wheel: Scroll image
• Middle Mouse: Pan image

⌨️ Keyboard Shortcuts:
• Ctrl+Z: Undo last action
• Ctrl+Y: Redo action
• ESC: Cancel current polygon
• Delete: Delete selected item

📏 Annotation Tips:
• Hold Ctrl while scrolling to zoom
• Zoom in for precise hole annotation
• Use perspective points for tilted plates
• Mark knobs completely for accurate size comparison

✅ Quality Guidelines:
• Holes: Should be circular and unblocked
• Text: Should be white and readable
• Knobs: Plus terminal must be larger than minus
• Surface: Check for defects and discoloration
        """
        messagebox.showinfo("Quick Guide", help_text)
    
    def save_state(self, action_type, data):
        """Save current state for undo/redo"""
        import time
        state = AnnotationState(
            action_type=action_type,
            data=data,
            timestamp=time.time()
        )
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Clear redo stack on new action
        self.status_var.set(f"Action: {action_type}")
    
    def undo(self):
        """Undo last action"""
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return
        
        state = self.undo_stack.pop()
        self.redo_stack.append(state)
        
        # Apply undo based on action type
        if state.action_type == "add_hole":
            if self.current_annotation.hole_polygons:
                self.current_annotation.hole_polygons.pop()
                self.current_annotation.hole_qualities.pop()
                self.update_hole_list()
        
        elif state.action_type == "set_text":
            self.current_annotation.text_polygon = None
            self.current_annotation.text_obb = None
        
        elif state.action_type == "set_plus_knob":
            self.current_annotation.plus_knob_polygon = None
            self.current_annotation.plus_knob_area = None
        
        elif state.action_type == "set_minus_knob":
            self.current_annotation.minus_knob_polygon = None
            self.current_annotation.minus_knob_area = None
        
        self.update_display()
        self.status_var.set(f"Undid: {state.action_type}")
    
    def redo(self):
        """Redo action"""
        if not self.redo_stack:
            self.status_var.set("Nothing to redo")
            return
        
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        
        # Apply redo based on action type
        if state.action_type == "add_hole":
            polygon, quality = state.data
            self.current_annotation.hole_polygons.append(polygon)
            self.current_annotation.hole_qualities.append(quality)
            self.update_hole_list()
        
        elif state.action_type == "set_text":
            self.current_annotation.text_polygon = state.data
            self.current_annotation.text_obb = self.polygon_to_obb(state.data)
        
        elif state.action_type == "set_plus_knob":
            self.current_annotation.plus_knob_polygon = state.data
            self.current_annotation.plus_knob_area = self.calculate_polygon_area(state.data)
        
        elif state.action_type == "set_minus_knob":
            self.current_annotation.minus_knob_polygon = state.data
            self.current_annotation.minus_knob_area = self.calculate_polygon_area(state.data)
        
        self.update_display()
        self.status_var.set(f"Redid: {state.action_type}")
    
    def on_zoom_changed(self, event=None):
        """Handle zoom change event"""
        zoom_percent = int(self.canvas.zoom_level * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.canvas.reset_zoom()
    
    def fit_to_window(self):
        """Fit image to window"""
        if self.current_image is None:
            return
        
        # Calculate scale to fit
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = self.current_image.shape[:2]
        
        # Ensure we have valid canvas dimensions
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 900, 700  # Default fallback
        
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)  # Don't zoom larger than 100%
        
        # Set zoom level and update display
        self.canvas.zoom_level = scale
        self.update_display()
        self.on_zoom_changed()
    
    def change_mode(self):
        """Change annotation mode"""
        self.annotation_mode = self.mode_var.get()
        self.current_polygon = []
        self.drawing = False
        self.update_display()
        self.status_var.set(f"Mode: {self.annotation_mode.replace('_', ' ').title()}")
    
    def on_canvas_click(self, event):
        """Handle canvas click with zoom support"""
        if self.current_image is None:
            return
        
        # Convert to image coordinates
        img_x, img_y = self.canvas.canvas_to_image_coords(event.x, event.y)
        
        # Ensure coordinates are within image bounds
        h, w = self.current_image.shape[:2]
        img_x = max(0, min(img_x, w-1))
        img_y = max(0, min(img_y, h-1))
        
        if self.annotation_mode in ["hole_polygon", "text_region", "plus_knob", "minus_knob"]:
            self.current_polygon.append((img_x, img_y))
            self.drawing = True
            self.update_display()
            
        elif self.annotation_mode == "perspective":
            if len(self.current_polygon) < 4:
                self.current_polygon.append((img_x, img_y))
                self.update_display()
                if len(self.current_polygon) == 4:
                    self.current_annotation.perspective_points = self.current_polygon.copy()
                    self.current_polygon = []
                    self.save_state("set_perspective", self.current_annotation.perspective_points)
                    self.status_var.set("Perspective points set")
    
    def on_double_click(self, event):
        """Close polygon on double click"""
        if self.drawing and len(self.current_polygon) >= 3:
            self.close_current_polygon()
    
    def on_canvas_motion(self, event):
        """Show preview with zoom support"""
        if self.drawing and self.current_polygon:
            # Only redraw if we have moved significantly to avoid flicker
            self.update_display()
            
            # Draw preview line
            if len(self.current_polygon) > 0:
                img_x, img_y = self.canvas.canvas_to_image_coords(event.x, event.y)
                last_x, last_y = self.current_polygon[-1]
                
                # Convert to canvas coordinates (zoomed coordinates)
                x1, y1 = self.canvas.image_to_canvas_coords(last_x, last_y)
                x2, y2 = self.canvas.image_to_canvas_coords(img_x, img_y)
                
                self.canvas.create_line(x1, y1, x2, y2, fill="yellow", 
                                      width=2, tags="preview", dash=(5, 5))
    
    def close_current_polygon(self):
        """Close and save current polygon"""
        if len(self.current_polygon) < 3:
            messagebox.showwarning("Warning", "Polygon needs at least 3 points")
            return
        
        if self.annotation_mode == "hole_polygon":
            self.current_annotation.hole_polygons.append(self.current_polygon.copy())
            self.current_annotation.hole_qualities.append("good")
            self.save_state("add_hole", (self.current_polygon.copy(), "good"))
            self.update_hole_list()
            
        elif self.annotation_mode == "text_region":
            self.current_annotation.text_polygon = self.current_polygon.copy()
            self.current_annotation.text_obb = self.polygon_to_obb(self.current_polygon)
            self.save_state("set_text", self.current_polygon.copy())
            
        elif self.annotation_mode == "plus_knob":
            self.current_annotation.plus_knob_polygon = self.current_polygon.copy()
            self.current_annotation.plus_knob_area = self.calculate_polygon_area(self.current_polygon)
            self.save_state("set_plus_knob", self.current_polygon.copy())
            
        elif self.annotation_mode == "minus_knob":
            self.current_annotation.minus_knob_polygon = self.current_polygon.copy()
            self.current_annotation.minus_knob_area = self.calculate_polygon_area(self.current_polygon)
            self.save_state("set_minus_knob", self.current_polygon.copy())
        
        self.current_polygon = []
        self.drawing = False
        self.update_display()
    
    def on_right_click(self, event):
        """Remove last point"""
        if self.drawing and self.current_polygon:
            self.current_polygon.pop()
            self.update_display()
    
    def cancel_current_polygon(self):
        """Cancel current polygon"""
        self.current_polygon = []
        self.drawing = False
        self.update_display()
        self.status_var.set("Polygon cancelled")
    
    def delete_selected(self):
        """Delete selected item"""
        if self.selected_item:
            if self.selected_item[0] == "hole":
                index = self.selected_item[1]
                if 0 <= index < len(self.current_annotation.hole_polygons):
                    removed_polygon = self.current_annotation.hole_polygons.pop(index)
                    removed_quality = self.current_annotation.hole_qualities.pop(index)
                    self.save_state("delete_hole", (index, removed_polygon, removed_quality))
                    self.update_hole_list()
                    self.selected_item = None
                    self.update_display()
    
    def polygon_to_obb(self, polygon):
        """Convert polygon to oriented bounding box"""
        points = np.array(polygon, dtype=np.float32)
        if len(points) < 3:
            return None
        
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
        """Calculate polygon area"""
        if len(polygon) < 3:
            return 0
        poly = ShapelyPolygon(polygon)
        return poly.area
    
    def update_hole_list(self):
        """Update hole list display"""
        self.hole_list.delete(0, tk.END)
        for i, (polygon, quality) in enumerate(zip(self.current_annotation.hole_polygons,
                                                   self.current_annotation.hole_qualities)):
            area = self.calculate_polygon_area(polygon)
            color_emoji = {"good": "🟢", "deformed": "🟡", "blocked": "🔴"}.get(quality, "⚪")
            self.hole_list.insert(tk.END, f"{color_emoji} Hole {i+1}: {quality} (area: {area:.0f})")
    
    def on_hole_select(self, event):
        """Handle hole selection"""
        selection = self.hole_list.curselection()
        if selection:
            index = selection[0]
            self.selected_item = ("hole", index)
            quality = self.current_annotation.hole_qualities[index]
            self.hole_quality_var.set(quality)
            self.update_display()
    
    def update_hole_quality(self):
        """Update selected hole quality"""
        if self.selected_item and self.selected_item[0] == "hole":
            index = self.selected_item[1]
            old_quality = self.current_annotation.hole_qualities[index]
            new_quality = self.hole_quality_var.get()
            self.current_annotation.hole_qualities[index] = new_quality
            self.save_state("change_hole_quality", (index, old_quality, new_quality))
            self.update_hole_list()
            self.update_display()
    
    def analyze_hole_circularity(self):
        """Analyze circularity of all holes"""
        if not self.current_annotation.hole_polygons:
            messagebox.showinfo("Info", "No holes annotated")
            return
        
        results = []
        for i, polygon in enumerate(self.current_annotation.hole_polygons):
            if len(polygon) < 3:
                continue
            
            poly = ShapelyPolygon(polygon)
            area = poly.area
            perimeter = poly.length
            
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter ** 2)
                results.append((i, circularity))
                
                # Auto-update quality based on circularity
                if circularity < 0.7:
                    self.current_annotation.hole_qualities[i] = "deformed"
                elif circularity < 0.85:
                    if self.current_annotation.hole_qualities[i] == "good":
                        self.current_annotation.hole_qualities[i] = "deformed"
        
        # Show results
        result_text = "Hole Circularity Analysis\n"
        result_text += "(1.0 = perfect circle, <0.7 = severely deformed)\n\n"
        
        for i, circ in sorted(results, key=lambda x: x[1]):
            quality = self.current_annotation.hole_qualities[i]
            emoji = {"good": "🟢", "deformed": "🟡", "blocked": "🔴"}.get(quality, "⚪")
            result_text += f"{emoji} Hole {i+1}: {circ:.3f} ({quality})\n"
        
        self.update_hole_list()
        self.update_display()
        
        messagebox.showinfo("Circularity Analysis", result_text)
    
    def calculate_knob_sizes(self):
        """Calculate and display knob sizes"""
        info_text = "Knob Size Analysis\n" + "="*30 + "\n\n"
        
        if self.current_annotation.plus_knob_polygon:
            plus_area = self.calculate_polygon_area(self.current_annotation.plus_knob_polygon)
            self.current_annotation.plus_knob_area = plus_area
            info_text += f"➕ Plus Terminal:\n"
            info_text += f"   Area: {plus_area:.1f} px²\n"
            info_text += f"   Approx. diameter: {2*math.sqrt(plus_area/math.pi):.1f} px\n\n"
        else:
            info_text += "➕ Plus Terminal: Not annotated\n\n"
        
        if self.current_annotation.minus_knob_polygon:
            minus_area = self.calculate_polygon_area(self.current_annotation.minus_knob_polygon)
            self.current_annotation.minus_knob_area = minus_area
            info_text += f"➖ Minus Terminal:\n"
            info_text += f"   Area: {minus_area:.1f} px²\n"
            info_text += f"   Approx. diameter: {2*math.sqrt(minus_area/math.pi):.1f} px\n\n"
        else:
            info_text += "➖ Minus Terminal: Not annotated\n\n"
        
        # Check sizing
        if (self.current_annotation.plus_knob_area and 
            self.current_annotation.minus_knob_area):
            
            if self.current_annotation.plus_knob_area > self.current_annotation.minus_knob_area:
                info_text += "✅ CORRECT: Plus terminal is larger than minus\n"
                self.current_annotation.knob_size_correct = True
                self.knob_correct_var.set(True)
            else:
                info_text += "❌ INCORRECT: Plus terminal should be larger!\n"
                self.current_annotation.knob_size_correct = False
                self.knob_correct_var.set(False)
            
            ratio = self.current_annotation.plus_knob_area / self.current_annotation.minus_knob_area
            info_text += f"\nArea ratio (Plus/Minus): {ratio:.3f}"
        
        self.knob_info_text.delete(1.0, tk.END)
        self.knob_info_text.insert(1.0, info_text)
        
        # Update knob quality
        self.update_knob_quality()
    
    def update_component_quality(self, component):
        """Update component quality assessment"""
        quality_var = getattr(self, f"{component.lower()}_quality_var")
        quality = quality_var.get()
        
        if component == "Holes":
            self.current_annotation.hole_quality = quality
        elif component == "Text":
            self.current_annotation.text_quality = quality
        elif component == "Knobs":
            self.current_annotation.knob_quality = quality
        elif component == "Surface":
            self.current_annotation.surface_quality = quality
        
        self.status_var.set(f"{component} quality set to: {quality}")
    
    def update_knob_quality(self):
        """Update knob quality based on size comparison"""
        if self.knob_correct_var.get():
            self.current_annotation.knob_size_correct = True
            self.knobs_quality_var.set("GOOD")
            self.current_annotation.knob_quality = "GOOD"
        else:
            self.current_annotation.knob_size_correct = False
            if (self.current_annotation.plus_knob_polygon and 
                self.current_annotation.minus_knob_polygon):
                self.knobs_quality_var.set("BAD")
                self.current_annotation.knob_quality = "BAD"
    
    def calculate_overall_quality(self):
        """Calculate overall quality based on components"""
        qualities = [
            self.current_annotation.hole_quality,
            self.current_annotation.text_quality,
            self.current_annotation.knob_quality,
            self.current_annotation.surface_quality
        ]
        
        # Count quality assessments
        good_count = qualities.count("GOOD")
        bad_count = qualities.count("BAD")
        unknown_count = qualities.count("UNKNOWN")
        
        # Determine overall quality
        if bad_count > 0:
            overall = "BAD"
        elif unknown_count > 1:
            overall = "UNKNOWN"
        elif good_count == 4:
            overall = "GOOD"
        else:
            overall = "UNKNOWN"
        
        self.overall_var.set(overall)
        self.current_annotation.overall_quality = overall
        
        # Show breakdown
        breakdown = f"Quality Breakdown:\n"
        breakdown += f"  Holes: {self.current_annotation.hole_quality}\n"
        breakdown += f"  Text: {self.current_annotation.text_quality}\n"
        breakdown += f"  Knobs: {self.current_annotation.knob_quality}\n"
        breakdown += f"  Surface: {self.current_annotation.surface_quality}\n"
        breakdown += f"\nOverall: {overall}"
        
        messagebox.showinfo("Quality Assessment", breakdown)
    
    def update_display(self):
        """Update canvas display with zoom support"""
        if self.current_image is None:
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Create display image with annotations
        display_image = self.current_image.copy()
        h, w = display_image.shape[:2]
        
        # Draw annotations at original scale
        overlay = display_image.copy()
        
        # Draw holes with quality-based colors
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
            
            # Highlight selected
            if self.selected_item and self.selected_item[0] == "hole" and self.selected_item[1] == i:
                cv2.polylines(display_image, [points], True, (255, 255, 0), 4)
            
            # Add label
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_image, f"{i+1}", (cx-10, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw text region
        if self.current_annotation.text_polygon:
            points = np.array(self.current_annotation.text_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (0, 255, 255))
            cv2.polylines(display_image, [points], True, (0, 255, 255), 3)
        
        # Draw knobs
        if self.current_annotation.plus_knob_polygon:
            points = np.array(self.current_annotation.plus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (255, 0, 255))
            cv2.polylines(display_image, [points], True, (255, 0, 255), 3)
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_image, "+", (cx-10, cy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        
        if self.current_annotation.minus_knob_polygon:
            points = np.array(self.current_annotation.minus_knob_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [points], (0, 255, 255))
            cv2.polylines(display_image, [points], True, (0, 255, 255), 3)
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_image, "-", (cx-5, cy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Draw perspective points
        if self.current_annotation.perspective_points:
            for i, point in enumerate(self.current_annotation.perspective_points):
                cv2.circle(display_image, point, 8, (255, 128, 0), -1)
                cv2.putText(display_image, f"P{i+1}", (point[0]+10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
        
        # Draw current polygon
        if self.current_polygon:
            points = np.array(self.current_polygon, dtype=np.int32)
            cv2.polylines(display_image, [points], False, (255, 255, 0), 2)
            for point in self.current_polygon:
                cv2.circle(display_image, point, 4, (255, 255, 0), -1)
        
        # Apply overlay
        cv2.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
        
        # Add quality indicators
        quality_y = 30
        for component, quality in [
            ("H", self.current_annotation.hole_quality),
            ("T", self.current_annotation.text_quality),
            ("K", self.current_annotation.knob_quality),
            ("S", self.current_annotation.surface_quality),
            ("O", self.current_annotation.overall_quality)
        ]:
            color = {
                "GOOD": (0, 255, 0),
                "BAD": (0, 0, 255),
                "UNKNOWN": (128, 128, 128)
            }.get(quality, (128, 128, 128))
            
            cv2.rectangle(display_image, (10, quality_y-20), (35, quality_y), color, -1)
            cv2.putText(display_image, component, (15, quality_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            quality_y += 30
        
        # Now apply zoom by resizing the image
        zoom_w = int(w * self.canvas.zoom_level)
        zoom_h = int(h * self.canvas.zoom_level)
        
        if self.canvas.zoom_level != 1.0:
            display_image = cv2.resize(display_image, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PhotoImage and display
        image_pil = Image.fromarray(display_image)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Create image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update scroll region to match zoomed image size
        self.canvas.configure(scrollregion=(0, 0, zoom_w, zoom_h))
    
    def load_images(self):
        """Load images from directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mov']
            self.image_list = []
            for ext in image_extensions:
                self.image_list.extend(Path(directory).glob(f"*{ext}"))
                self.image_list.extend(Path(directory).glob(f"*{ext.upper()}"))
            
            self.image_list = sorted(list(set(self.image_list)))
            
            if self.image_list:
                self.current_index = 0
                self.load_current_image()
                messagebox.showinfo("Success", f"Loaded {len(self.image_list)} images")
            else:
                messagebox.showwarning("Warning", "No images found")
    
    def load_current_image(self):
        """Load current image"""
        if not self.image_list:
            return
        
        self.current_image_path = str(self.image_list[self.current_index])
        
        # Handle video files
        if self.current_image_path.lower().endswith('.mov'):
            # Extract first frame
            cap = cv2.VideoCapture(self.current_image_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                messagebox.showerror("Error", "Could not read video file")
                return
        else:
            image = cv2.imread(self.current_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.current_image = image
        
        # Load or create annotation
        self.load_annotation()
        
        # Reset zoom and fit to window
        self.canvas.reset_zoom()
        
        # Update UI
        self.update_display()
        self.update_hole_list()
        self.image_label.config(text=f"{self.current_index + 1}/{len(self.image_list)}")
        
        # Clear undo/redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()
    
    def load_annotation(self):
        """Load existing annotation"""
        base_path = self.current_image_path.rsplit('.', 1)[0]
        annotation_path = f"{base_path}_enhanced_annotation.json"
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                self.current_annotation = EnhancedAnnotation(**data)
                
                # Update UI
                self.holes_quality_var.set(self.current_annotation.hole_quality)
                self.text_quality_var.set(self.current_annotation.text_quality)
                self.knobs_quality_var.set(self.current_annotation.knob_quality)
                self.surface_quality_var.set(self.current_annotation.surface_quality)
                self.overall_var.set(self.current_annotation.overall_quality)
                self.confidence_var.set(self.current_annotation.confidence_score)
                self.knob_correct_var.set(self.current_annotation.knob_size_correct)
                
                self.notes_text.delete(1.0, tk.END)
                self.notes_text.insert(1.0, self.current_annotation.notes)
        else:
            # Create new annotation
            h, w = self.current_image.shape[:2]
            self.current_annotation = EnhancedAnnotation(
                image_path=self.current_image_path,
                image_width=w,
                image_height=h
            )
    
    def save_annotations(self):
        """Save current annotations"""
        if not self.current_annotation:
            return
        
        # Update from UI
        self.current_annotation.hole_quality = self.holes_quality_var.get()
        self.current_annotation.text_quality = self.text_quality_var.get()
        self.current_annotation.knob_quality = self.knobs_quality_var.get()
        self.current_annotation.surface_quality = self.surface_quality_var.get()
        self.current_annotation.overall_quality = self.overall_var.get()
        self.current_annotation.confidence_score = self.confidence_var.get()
        self.current_annotation.knob_size_correct = self.knob_correct_var.get()
        self.current_annotation.notes = self.notes_text.get(1.0, tk.END).strip()
        
        # Save
        base_path = self.current_image_path.rsplit('.', 1)[0]
        annotation_path = f"{base_path}_enhanced_annotation.json"
        
        with open(annotation_path, 'w') as f:
            json.dump(asdict(self.current_annotation), f, indent=2)
        
        self.status_var.set("Annotations saved!")
    
    def previous_image(self):
        """Go to previous image"""
        if self.image_list and self.current_index > 0:
            self.save_annotations()
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.save_annotations()
            self.current_index += 1
            self.load_current_image()
    
    def clear_all(self):
        """Clear all annotations"""
        if messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            h, w = self.current_image.shape[:2]
            self.current_annotation = EnhancedAnnotation(
                image_path=self.current_image_path,
                image_width=w,
                image_height=h
            )
            
            # Reset UI
            for var in [self.holes_quality_var, self.text_quality_var, 
                       self.knobs_quality_var, self.surface_quality_var, 
                       self.overall_var]:
                var.set("UNKNOWN")
            
            self.confidence_var.set(1.0)
            self.knob_correct_var.set(False)
            self.notes_text.delete(1.0, tk.END)
            self.hole_list.delete(0, tk.END)
            
            self.update_display()
            self.undo_stack.clear()
            self.redo_stack.clear()
    
    def export_training_data(self):
        """Export annotations for training"""
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        # Create directories
        for subdir in ["images", "masks", "annotations"]:
            os.makedirs(os.path.join(export_dir, subdir), exist_ok=True)
        
        # Process all annotations
        exported_count = 0
        all_annotations = []
        
        for img_path in self.image_list:
            base_path = str(img_path).rsplit('.', 1)[0]
            ann_path = f"{base_path}_enhanced_annotation.json"
            
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)
                    all_annotations.append(ann_data)
                
                # Copy image
                img_name = Path(img_path).name
                if img_name.lower().endswith('.mov'):
                    # Extract frame for video
                    cap = cv2.VideoCapture(str(img_path))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        img_name = img_name.rsplit('.', 1)[0] + '.jpg'
                        cv2.imwrite(os.path.join(export_dir, "images", img_name), frame)
                else:
                    import shutil
                    shutil.copy(str(img_path), os.path.join(export_dir, "images", img_name))
                
                exported_count += 1
        
        # Save consolidated annotations
        with open(os.path.join(export_dir, "all_annotations.json"), 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        # Create summary statistics
        stats = {
            "total_images": len(all_annotations),
            "quality_distribution": {
                "overall": {"GOOD": 0, "BAD": 0, "UNKNOWN": 0},
                "holes": {"GOOD": 0, "BAD": 0, "UNKNOWN": 0},
                "text": {"GOOD": 0, "BAD": 0, "UNKNOWN": 0},
                "knobs": {"GOOD": 0, "BAD": 0, "UNKNOWN": 0},
                "surface": {"GOOD": 0, "BAD": 0, "UNKNOWN": 0}
            },
            "knob_sizing": {
                "correct": sum(1 for a in all_annotations if a.get('knob_size_correct', False)),
                "incorrect": sum(1 for a in all_annotations if not a.get('knob_size_correct', False) 
                               and a.get('plus_knob_polygon') and a.get('minus_knob_polygon'))
            },
            "average_holes_per_image": np.mean([len(a.get('hole_polygons', [])) for a in all_annotations]),
            "confidence_scores": {
                "mean": np.mean([a.get('confidence_score', 1.0) for a in all_annotations]),
                "min": np.min([a.get('confidence_score', 1.0) for a in all_annotations]),
                "max": np.max([a.get('confidence_score', 1.0) for a in all_annotations])
            }
        }
        
        # Count quality distributions
        for ann in all_annotations:
            for component in ['overall', 'hole', 'text', 'knob', 'surface']:
                quality_key = f"{component}_quality"
                quality = ann.get(quality_key, "UNKNOWN")
                if component in stats["quality_distribution"]:
                    stats["quality_distribution"][component][quality] += 1
                else:
                    stats["quality_distribution"][component.rstrip('_quality')][quality] += 1
        
        # Save statistics
        with open(os.path.join(export_dir, "dataset_statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate report
        report = f"Export Summary\n{'='*50}\n\n"
        report += f"Total images exported: {exported_count}\n\n"
        report += "Quality Distribution:\n"
        for component, dist in stats["quality_distribution"].items():
            report += f"  {component.title()}:\n"
            for quality, count in dist.items():
                report += f"    {quality}: {count}\n"
        report += f"\nKnob Sizing:\n"
        report += f"  Correct: {stats['knob_sizing']['correct']}\n"
        report += f"  Incorrect: {stats['knob_sizing']['incorrect']}\n"
        report += f"\nAverage holes per image: {stats['average_holes_per_image']:.1f}\n"
        report += f"\nConfidence scores:\n"
        report += f"  Mean: {stats['confidence_scores']['mean']:.2f}\n"
        report += f"  Range: {stats['confidence_scores']['min']:.2f} - {stats['confidence_scores']['max']:.2f}\n"
        
        with open(os.path.join(export_dir, "export_report.txt"), 'w') as f:
            f.write(report)
        
        messagebox.showinfo("Export Complete", f"Exported {exported_count} annotations\n\nSee export_report.txt for details")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedBatteryLabelingTool(root)
    root.mainloop()