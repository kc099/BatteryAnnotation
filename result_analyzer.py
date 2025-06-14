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
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(results_frame, text="Statistics", padding="10")
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
        charts_frame = ttk.LabelFrame(results_frame, text="Distribution Charts", padding="10")
        charts_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.rowconfigure(0, weight=1)
        
        # Initial empty charts
        self.clear_charts()
    
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
        
        # Generate statistics and charts
        self.generate_statistics()
        self.generate_charts()
        
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
        
        # Component-specific analysis
        hole_eccentricities = []
        knob_ratios = []
        text_white_ratios = []
        text_contrasts = []
        
        for result in self.results_data:
            details = result['analysis_details']
            
            if 'hole_analysis' in details:
                hole_eccentricities.append(details['hole_analysis']['eccentricity'])
            
            if 'knob_analysis' in details:
                knob_ratios.append(details['knob_analysis']['area_ratio'])
            
            if 'text_analysis' in details:
                text_white_ratios.append(details['text_analysis']['white_score'])
                # Extract contrast from status string
                status = details['text_analysis']['status']
                try:
                    contrast_part = status.split('Contrast: ')[1].split(' ')[0]
                    text_contrasts.append(float(contrast_part))
                except:
                    pass
        
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
   • Min:  {np.min(hole_eccentricities):.3f}
   • Max:  {np.max(hole_eccentricities):.3f}
   • Threshold: 0.400

🔘 KNOB SIZE QUALITY:
   Good: {knob_good} ({knob_good/total_images*100:.1f}%)
   Bad:  {total_images-knob_good} ({(total_images-knob_good)/total_images*100:.1f}%)
   
   Area Ratio Statistics:
   • Mean: {np.mean(knob_ratios):.3f}
   • Std:  {np.std(knob_ratios):.3f}
   • Min:  {np.min(knob_ratios):.3f}
   • Max:  {np.max(knob_ratios):.3f}
   • Threshold: 1.200

📝 TEXT COLOR QUALITY:
   Good: {text_good} ({text_good/total_images*100:.1f}%)
   Bad:  {total_images-text_good} ({(total_images-text_good)/total_images*100:.1f}%)
   
   White Ratio Statistics:
   • Mean: {np.mean(text_white_ratios):.3f}
   • Std:  {np.std(text_white_ratios):.3f}
   • Min:  {np.min(text_white_ratios):.3f}
   • Max:  {np.max(text_white_ratios):.3f}
   • Threshold: 0.010
   
   Contrast Statistics:
   • Mean: {np.mean(text_contrasts):.1f}
   • Std:  {np.std(text_contrasts):.1f}
   • Min:  {np.min(text_contrasts):.1f}
   • Max:  {np.max(text_contrasts):.1f}
   • Threshold: 20.0

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

def main():
    root = tk.Tk()
    app = ResultAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 