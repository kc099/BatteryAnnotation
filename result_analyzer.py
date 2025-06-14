#!/usr/bin/env python3
"""
Battery Quality Result Analyzer GUI

Analyzes inference results from a folder and shows distribution statistics.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ResultAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Quality Result Analyzer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.results_data = []
        self.folder_path = None
        
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
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Battery Quality Result Analyzer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Folder selection frame
        folder_frame = ttk.LabelFrame(main_frame, text="Select Results Folder", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Button(folder_frame, text="Browse Folder", 
                  command=self.browse_folder).grid(row=0, column=0, padx=(0, 10))
        
        self.folder_label = ttk.Label(folder_frame, text="No folder selected", 
                                     foreground="gray")
        self.folder_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Button(folder_frame, text="Analyze Results", 
                  command=self.analyze_results).grid(row=0, column=2, padx=(10, 0))
        
        # Results frame with notebook for tabs
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Statistics and Charts
        stats_charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_charts_frame, text="Statistics & Charts")
        
        stats_charts_frame.columnconfigure(0, weight=1)
        stats_charts_frame.columnconfigure(1, weight=1)
        stats_charts_frame.rowconfigure(0, weight=1)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(stats_charts_frame, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create text widget for statistics
        self.stats_text = tk.Text(stats_frame, width=40, height=25, font=("Consolas", 10))
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        # Charts panel
        charts_frame = ttk.LabelFrame(stats_charts_frame, text="Distribution Charts", padding="10")
        charts_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Tab 2: Detailed Data Table
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="Detailed Data")
        
        # Create table with sorting and filtering options
        self.setup_data_table(table_frame)
        
        # Initial empty charts
        self.clear_charts()
    
    def setup_data_table(self, parent_frame):
        """Setup the detailed data table"""
        # Control frame for filtering and sorting
        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        control_frame.columnconfigure(1, weight=1)
        
        # Filter options
        ttk.Label(control_frame, text="Filter:").grid(row=0, column=0, padx=(0, 5))
        
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var, 
                                   values=["All", "Good Overall", "Bad Overall", "Good Hole", "Bad Hole", 
                                          "Good Text", "Bad Text", "Good Knob", "Bad Knob"], 
                                   state="readonly", width=15)
        filter_combo.grid(row=0, column=1, padx=(0, 10), sticky=tk.W)
        filter_combo.bind('<<ComboboxSelected>>', self.filter_table)
        
        # Sort options
        ttk.Label(control_frame, text="Sort by:").grid(row=0, column=2, padx=(10, 5))
        
        self.sort_var = tk.StringVar(value="Filename")
        sort_combo = ttk.Combobox(control_frame, textvariable=self.sort_var,
                                 values=["Filename", "Overall Quality", "Hole Eccentricity", 
                                        "Knob Ratio", "Text Score", "Detections"], 
                                 state="readonly", width=15)
        sort_combo.grid(row=0, column=3, padx=(0, 10))
        sort_combo.bind('<<ComboboxSelected>>', self.sort_table)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=self.refresh_table).grid(row=0, column=4, padx=(10, 0))
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(parent_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Define columns
        columns = ("Filename", "Overall", "Hole", "Text", "Knob", "Hole_Ecc", "Knob_Ratio", "Text_Score", "Detections", "Text_Status")
        
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Define column headings and widths
        column_config = {
            "Filename": ("File Name", 200),
            "Overall": ("Overall", 80),
            "Hole": ("Hole", 60),
            "Text": ("Text", 60),
            "Knob": ("Knob", 60),
            "Hole_Ecc": ("Hole Ecc.", 80),
            "Knob_Ratio": ("Knob Ratio", 80),
            "Text_Score": ("Text Score", 80),
            "Detections": ("Objects", 70),
            "Text_Status": ("Text Analysis", 300)
        }
        
        for col, (heading, width) in column_config.items():
            self.tree.heading(col, text=heading, command=lambda c=col: self.sort_by_column(c))
            self.tree.column(col, width=width, minwidth=50)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)
        
        # Store original data for filtering/sorting
        self.table_data = []
    
    def browse_folder(self):
        """Browse for results folder"""
        folder_path = filedialog.askdirectory(
            title="Select Inference Results Folder",
            initialdir=os.getcwd()
        )
        
        if folder_path:
            self.folder_path = Path(folder_path)
            self.folder_label.config(text=str(self.folder_path), foreground="black")
    
    def analyze_results(self):
        """Analyze all JSON results in the selected folder"""
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder first!")
            return
        
        # Find all JSON result files
        json_files = list(self.folder_path.glob("*_results.json"))
        
        if not json_files:
            messagebox.showerror("Error", f"No *_results.json files found in {self.folder_path}")
            return
        
        # Load and parse all results
        self.results_data = []
        failed_files = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.results_data.append(data)
            except Exception as e:
                failed_files.append(f"{json_file.name}: {str(e)}")
        
        if failed_files:
            messagebox.showwarning("Warning", 
                                 f"Failed to load {len(failed_files)} files:\n" + 
                                 "\n".join(failed_files[:5]) + 
                                 ("..." if len(failed_files) > 5 else ""))
        
        if not self.results_data:
            messagebox.showerror("Error", "No valid result files could be loaded!")
            return
        
        # Generate statistics, charts, and populate table
        self.generate_statistics()
        self.generate_charts()
        self.populate_table()
        
        messagebox.showinfo("Success", 
                           f"Analyzed {len(self.results_data)} result files successfully!")
    
    def generate_statistics(self):
        """Generate detailed statistics from the results"""
        if not self.results_data:
            return
        
        total_images = len(self.results_data)
        
        # Overall quality counts
        overall_good = sum(1 for r in self.results_data if r['overall_quality'] == 'GOOD')
        overall_bad = total_images - overall_good
        
        # Component quality counts
        hole_good = sum(1 for r in self.results_data if r['hole_good'])
        text_good = sum(1 for r in self.results_data if r['text_color_good'])
        knob_good = sum(1 for r in self.results_data if r['knob_size_good'])
        
        # Detection counts
        total_detections = sum(len(r['detections']['boxes']) for r in self.results_data)
        avg_detections = total_detections / total_images
        
        # Component-specific analysis with file tracking
        hole_data = []
        knob_data = []
        text_data = []
        
        for result in self.results_data:
            filename = result.get('image_name', 'Unknown')
            details = result['analysis_details']
            
            if 'hole_analysis' in details:
                hole_data.append((filename, details['hole_analysis']['eccentricity']))
            
            if 'knob_analysis' in details:
                knob_data.append((filename, details['knob_analysis']['area_ratio']))
            
            if 'text_analysis' in details:
                white_score = details['text_analysis']['white_score']
                text_data.append((filename, white_score))
        
        # Extract values for statistics
        hole_eccentricities = [data[1] for data in hole_data]
        knob_ratios = [data[1] for data in knob_data]
        text_white_ratios = [data[1] for data in text_data]
        
        # Find min/max files
        hole_min_file = min(hole_data, key=lambda x: x[1])[0] if hole_data else "N/A"
        hole_max_file = max(hole_data, key=lambda x: x[1])[0] if hole_data else "N/A"
        knob_min_file = min(knob_data, key=lambda x: x[1])[0] if knob_data else "N/A"
        knob_max_file = max(knob_data, key=lambda x: x[1])[0] if knob_data else "N/A"
        text_min_file = min(text_data, key=lambda x: x[1])[0] if text_data else "N/A"
        text_max_file = max(text_data, key=lambda x: x[1])[0] if text_data else "N/A"
        
        # Generate statistics text
        stats_text = f"""📊 BATTERY QUALITY ANALYSIS RESULTS
{'='*50}

📁 DATASET OVERVIEW:
   Total Images Analyzed: {total_images}
   Total Object Detections: {total_detections}
   Average Detections per Image: {avg_detections:.1f}

🎯 OVERALL QUALITY DISTRIBUTION:
   GOOD: {overall_good} ({overall_good/total_images*100:.1f}%)
   BAD:  {overall_bad} ({overall_bad/total_images*100:.1f}%)

🔍 COMPONENT ANALYSIS:

🕳️  HOLE QUALITY:
   Good: {hole_good} ({hole_good/total_images*100:.1f}%)
   Bad:  {total_images-hole_good} ({(total_images-hole_good)/total_images*100:.1f}%)
   
   Eccentricity Statistics:
   • Mean: {np.mean(hole_eccentricities):.3f}
   • Std:  {np.std(hole_eccentricities):.3f}
   • Min:  {np.min(hole_eccentricities):.3f} ({hole_min_file})
   • Max:  {np.max(hole_eccentricities):.3f} ({hole_max_file})
   • Threshold: 0.400

🔘 KNOB SIZE QUALITY:
   Good: {knob_good} ({knob_good/total_images*100:.1f}%)
   Bad:  {total_images-knob_good} ({(total_images-knob_good)/total_images*100:.1f}%)
   
   Area Ratio Statistics:
   • Mean: {np.mean(knob_ratios):.3f}
   • Std:  {np.std(knob_ratios):.3f}
   • Min:  {np.min(knob_ratios):.3f} ({knob_min_file})
   • Max:  {np.max(knob_ratios):.3f} ({knob_max_file})
   • Threshold: 1.200

📝 TEXT COLOR QUALITY:
   Good: {text_good} ({text_good/total_images*100:.1f}%)
   Bad:  {total_images-text_good} ({(total_images-text_good)/total_images*100:.1f}%)
   
   White Ratio Statistics:
   • Mean: {np.mean(text_white_ratios):.3f}
   • Std:  {np.std(text_white_ratios):.3f}
   • Min:  {np.min(text_white_ratios):.3f} ({text_min_file})
   • Max:  {np.max(text_white_ratios):.3f} ({text_max_file})
   • Threshold: 0.010

📈 QUALITY CORRELATIONS:
   All Good Components: {sum(1 for r in self.results_data if r['hole_good'] and r['text_color_good'] and r['knob_size_good'])} ({sum(1 for r in self.results_data if r['hole_good'] and r['text_color_good'] and r['knob_size_good'])/total_images*100:.1f}%)
   All Bad Components:  {sum(1 for r in self.results_data if not r['hole_good'] and not r['text_color_good'] and not r['knob_size_good'])} ({sum(1 for r in self.results_data if not r['hole_good'] and not r['text_color_good'] and not r['knob_size_good'])/total_images*100:.1f}%)

🔧 FAILURE MODES:
   Hole Only Bad:  {sum(1 for r in self.results_data if not r['hole_good'] and r['text_color_good'] and r['knob_size_good'])} images
   Text Only Bad:  {sum(1 for r in self.results_data if r['hole_good'] and not r['text_color_good'] and r['knob_size_good'])} images
   Knob Only Bad:  {sum(1 for r in self.results_data if r['hole_good'] and r['text_color_good'] and not r['knob_size_good'])} images
"""
        
        # Update statistics text widget
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def generate_charts(self):
        """Generate distribution charts"""
        if not self.results_data:
            return
        
        # Clear previous charts
        for ax in self.axes.flat:
            ax.clear()
        
        total_images = len(self.results_data)
        
        # 1. Overall Quality Pie Chart
        overall_good = sum(1 for r in self.results_data if r['overall_quality'] == 'GOOD')
        overall_bad = total_images - overall_good
        
        self.axes[0, 0].pie([overall_good, overall_bad], 
                           labels=['GOOD', 'BAD'], 
                           colors=['#2ecc71', '#e74c3c'],
                           autopct='%1.1f%%',
                           startangle=90)
        self.axes[0, 0].set_title('Overall Quality Distribution')
        
        # 2. Component Quality Bar Chart
        hole_good = sum(1 for r in self.results_data if r['hole_good'])
        text_good = sum(1 for r in self.results_data if r['text_color_good'])
        knob_good = sum(1 for r in self.results_data if r['knob_size_good'])
        
        components = ['Hole', 'Text', 'Knob']
        good_counts = [hole_good, text_good, knob_good]
        bad_counts = [total_images - hole_good, total_images - text_good, total_images - knob_good]
        
        x = np.arange(len(components))
        width = 0.35
        
        self.axes[0, 1].bar(x - width/2, good_counts, width, label='Good', color='#2ecc71')
        self.axes[0, 1].bar(x + width/2, bad_counts, width, label='Bad', color='#e74c3c')
        
        self.axes[0, 1].set_xlabel('Components')
        self.axes[0, 1].set_ylabel('Count')
        self.axes[0, 1].set_title('Component Quality Distribution')
        self.axes[0, 1].set_xticks(x)
        self.axes[0, 1].set_xticklabels(components)
        self.axes[0, 1].legend()
        
        # 3. Hole Eccentricity Histogram
        hole_eccentricities = [r['analysis_details']['hole_analysis']['eccentricity'] 
                              for r in self.results_data if 'hole_analysis' in r['analysis_details']]
        
        self.axes[1, 0].hist(hole_eccentricities, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        self.axes[1, 0].axvline(x=0.4, color='red', linestyle='--', label='Threshold (0.4)')
        self.axes[1, 0].set_xlabel('Eccentricity')
        self.axes[1, 0].set_ylabel('Frequency')
        self.axes[1, 0].set_title('Hole Eccentricity Distribution')
        self.axes[1, 0].legend()
        
        # 4. Text White Ratio Histogram
        text_white_ratios = [r['analysis_details']['text_analysis']['white_score'] 
                            for r in self.results_data if 'text_analysis' in r['analysis_details']]
        
        self.axes[1, 1].hist(text_white_ratios, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        self.axes[1, 1].axvline(x=0.01, color='red', linestyle='--', label='Threshold (0.01)')
        self.axes[1, 1].set_xlabel('White Ratio')
        self.axes[1, 1].set_ylabel('Frequency')
        self.axes[1, 1].set_title('Text White Ratio Distribution')
        self.axes[1, 1].legend()
        
        # Refresh canvas
        self.canvas.draw()
    
    def clear_charts(self):
        """Clear all charts"""
        for ax in self.axes.flat:
            ax.clear()
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        self.canvas.draw()
    
    def populate_table(self):
        """Populate the data table with analysis results"""
        if not self.results_data:
            return
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.table_data = []
        
        for result in self.results_data:
            # Extract data
            filename = result.get('image_name', 'Unknown')
            overall_quality = result.get('overall_quality', 'Unknown')
            hole_good = 'GOOD' if result.get('hole_good', False) else 'BAD'
            text_good = 'GOOD' if result.get('text_color_good', False) else 'BAD'
            knob_good = 'GOOD' if result.get('knob_size_good', False) else 'BAD'
            
            # Get detailed analysis
            details = result.get('analysis_details', {})
            
            hole_ecc = details.get('hole_analysis', {}).get('eccentricity', 0.0)
            knob_ratio = details.get('knob_analysis', {}).get('area_ratio', 0.0)
            text_score = details.get('text_analysis', {}).get('white_score', 0.0)
            text_status = details.get('text_analysis', {}).get('status', 'Unknown')
            
            detections_count = len(result.get('detections', {}).get('boxes', []))
            
            # Create row data
            row_data = {
                'filename': filename,
                'overall': overall_quality,
                'hole': hole_good,
                'text': text_good,
                'knob': knob_good,
                'hole_ecc': hole_ecc,
                'knob_ratio': knob_ratio,
                'text_score': text_score,
                'detections': detections_count,
                'text_status': text_status
            }
            
            self.table_data.append(row_data)
            
            # Insert into treeview
            values = (
                filename,
                overall_quality,
                hole_good,
                text_good,
                knob_good,
                f"{hole_ecc:.3f}",
                f"{knob_ratio:.3f}",
                f"{text_score:.3f}",
                detections_count,
                text_status
            )
            
            # Color coding
            tags = []
            if overall_quality == 'GOOD':
                tags.append('good_overall')
            else:
                tags.append('bad_overall')
            
            self.tree.insert('', 'end', values=values, tags=tags)
        
        # Configure tags for color coding
        self.tree.tag_configure('good_overall', background='#d4edda')
        self.tree.tag_configure('bad_overall', background='#f8d7da')
    
    def filter_table(self, event=None):
        """Filter table based on selected criteria"""
        filter_value = self.filter_var.get()
        
        # Clear current items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Filter data
        filtered_data = []
        for row in self.table_data:
            include = True
            
            if filter_value == "Good Overall":
                include = row['overall'] == 'GOOD'
            elif filter_value == "Bad Overall":
                include = row['overall'] == 'BAD'
            elif filter_value == "Good Hole":
                include = row['hole'] == 'GOOD'
            elif filter_value == "Bad Hole":
                include = row['hole'] == 'BAD'
            elif filter_value == "Good Text":
                include = row['text'] == 'GOOD'
            elif filter_value == "Bad Text":
                include = row['text'] == 'BAD'
            elif filter_value == "Good Knob":
                include = row['knob'] == 'GOOD'
            elif filter_value == "Bad Knob":
                include = row['knob'] == 'BAD'
            
            if include:
                filtered_data.append(row)
        
        # Populate filtered data
        for row in filtered_data:
            values = (
                row['filename'],
                row['overall'],
                row['hole'],
                row['text'],
                row['knob'],
                f"{row['hole_ecc']:.3f}",
                f"{row['knob_ratio']:.3f}",
                f"{row['text_score']:.3f}",
                row['detections'],
                row['text_status']
            )
            
            tags = ['good_overall'] if row['overall'] == 'GOOD' else ['bad_overall']
            self.tree.insert('', 'end', values=values, tags=tags)
    
    def sort_table(self, event=None):
        """Sort table based on selected criteria"""
        sort_value = self.sort_var.get()
        
        # Define sort key mapping
        sort_keys = {
            "Filename": lambda x: x['filename'].lower(),
            "Overall Quality": lambda x: x['overall'],
            "Hole Eccentricity": lambda x: x['hole_ecc'],
            "Knob Ratio": lambda x: x['knob_ratio'],
            "Text Score": lambda x: x['text_score'],
            "Detections": lambda x: x['detections']
        }
        
        if sort_value in sort_keys:
            # Sort data
            sorted_data = sorted(self.table_data, key=sort_keys[sort_value])
            
            # Clear and repopulate
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for row in sorted_data:
                values = (
                    row['filename'],
                    row['overall'],
                    row['hole'],
                    row['text'],
                    row['knob'],
                    f"{row['hole_ecc']:.3f}",
                    f"{row['knob_ratio']:.3f}",
                    f"{row['text_score']:.3f}",
                    row['detections'],
                    row['text_status']
                )
                
                tags = ['good_overall'] if row['overall'] == 'GOOD' else ['bad_overall']
                self.tree.insert('', 'end', values=values, tags=tags)
    
    def sort_by_column(self, col):
        """Sort table by clicking column header"""
        # Map column names to sort keys
        column_mapping = {
            "Filename": "Filename",
            "Overall": "Overall Quality",
            "Hole_Ecc": "Hole Eccentricity",
            "Knob_Ratio": "Knob Ratio",
            "Text_Score": "Text Score",
            "Detections": "Detections"
        }
        
        if col in column_mapping:
            self.sort_var.set(column_mapping[col])
            self.sort_table()
    
    def refresh_table(self):
        """Refresh table with current data"""
        self.populate_table()

def main():
    root = tk.Tk()
    app = ResultAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 