import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import itertools

def load_annotations(data_folder):
    """
    Load all annotation files from the data folder and its subdirectories.
    Returns a list of annotation dictionaries and their file paths.
    """
    annotations = []
    annotation_files = []
    data_path = Path(data_folder)
    
    # Find all JSON annotation files
    json_files = list(data_path.rglob('*_enhanced_annotation.json'))
    
    print(f"Found {len(json_files)} annotation files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                ann = json.load(f)
                # Add file path to annotation for reference
                ann['_file_path'] = str(json_file)
                ann['_image_name'] = json_file.name.replace('_enhanced_annotation.json', '')
                annotations.append(ann)
                annotation_files.append(json_file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return annotations, annotation_files

def analyze_quality_distributions(annotations):
    """
    Analyze quality distributions from annotations.
    Returns a dictionary with quality statistics.
    """
    # Only consider these quality fields (excluding surface_quality)
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'overall_quality']
    
    # Initialize counters
    quality_stats = {}
    for field in quality_fields:
        quality_stats[field] = Counter()
    
    # Count quality values
    for ann in annotations:
        for field in quality_fields:
            if field in ann:
                quality_stats[field][ann[field]] += 1
    
    return quality_stats

def create_quality_visualizations(quality_stats, output_dir=None):
    """
    Create visualizations for quality distributions.
    """
    # Only consider these quality fields (excluding surface_quality)
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'overall_quality']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots (2x2 for 4 quality fields)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Quality Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, field in enumerate(quality_fields):
        ax = axes_flat[i]
        
        if field in quality_stats and quality_stats[field]:
            # Get data for this quality field
            data = quality_stats[field]
            labels = list(data.keys())
            values = list(data.values())
            
            # Create bar plot
            bars = ax.bar(labels, values, color=sns.color_palette("husl", len(labels)))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{field.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Count')
            ax.set_xlabel('Quality Rating')
            ax.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            total = sum(values)
            for j, (label, value) in enumerate(zip(labels, values)):
                percentage = (value / total) * 100
                ax.text(j, value/2, f'{percentage:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{field.replace("_", " ").title()}')
    
    # No need to remove subplot since we have exactly 4 quality fields for 2x2 grid
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'quality_distribution_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.show()

def create_summary_statistics(quality_stats):
    """
    Create a summary table of quality statistics.
    """
    # Only consider these quality fields (excluding surface_quality)
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'overall_quality']
    
    print("\n" + "="*80)
    print("DATASET QUALITY SUMMARY STATISTICS")
    print("="*80)
    
    summary_data = []
    
    for field in quality_fields:
        if field in quality_stats and quality_stats[field]:
            data = quality_stats[field]
            total = sum(data.values())
            
            # Calculate good vs bad distribution
            good_count = sum(count for quality, count in data.items() 
                           if quality.upper() in ['GOOD', 'EXCELLENT', 'PERFECT'])
            bad_count = sum(count for quality, count in data.items() 
                          if quality.upper() in ['BAD', 'POOR', 'DEFECTIVE', 'FAIL'])
            other_count = total - good_count - bad_count
            
            good_percent = (good_count / total) * 100 if total > 0 else 0
            bad_percent = (bad_count / total) * 100 if total > 0 else 0
            other_percent = (other_count / total) * 100 if total > 0 else 0
            
            summary_data.append({
                'Quality Field': field.replace('_', ' ').title(),
                'Total Samples': total,
                'Good': f"{good_count} ({good_percent:.1f}%)",
                'Bad': f"{bad_count} ({bad_percent:.1f}%)",
                'Other': f"{other_count} ({other_percent:.1f}%)",
                'Distribution': dict(data)
            })
            
            print(f"\n{field.replace('_', ' ').title()}:")
            print(f"  Total samples: {total}")
            print(f"  Good quality: {good_count} ({good_percent:.1f}%)")
            print(f"  Bad quality: {bad_count} ({bad_percent:.1f}%)")
            print(f"  Other quality: {other_count} ({other_percent:.1f}%)")
            print(f"  Full distribution: {dict(data)}")
    
    return summary_data

def analyze_defect_types(annotations):
    """
    Analyze defect types if available in annotations.
    """
    defect_counter = Counter()
    samples_with_defects = 0
    
    for ann in annotations:
        if 'defect_types' in ann and ann['defect_types']:
            samples_with_defects += 1
            for defect in ann['defect_types']:
                defect_counter[defect] += 1
    
    if defect_counter:
        print(f"\nDEFECT ANALYSIS:")
        print(f"  Samples with defects: {samples_with_defects}")
        print(f"  Total defect occurrences: {sum(defect_counter.values())}")
        print(f"  Defect types distribution:")
        for defect, count in defect_counter.most_common():
            print(f"    {defect}: {count}")
    
    return defect_counter

def analyze_quality_combinations(annotations):
    """
    Analyze combinations of quality issues across different components.
    """
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'surface_quality']
    
    # Initialize counters for different combinations
    combination_stats = {
        'hole_knob': {'good': [], 'bad': []},
        'knob_text': {'good': [], 'bad': []},
        'text_surface': {'good': [], 'bad': []},
        'hole_text': {'good': [], 'bad': []},
        'hole_surface': {'good': [], 'bad': []},
        'knob_surface': {'good': [], 'bad': []},
        'all_components': {'good': [], 'bad': []}
    }
    
    for ann in annotations:
        image_name = ann.get('_image_name', 'unknown')
        
        # Check if all quality fields exist
        if all(field in ann for field in quality_fields):
            hole_qual = ann['hole_quality'].upper()
            text_qual = ann['text_quality'].upper()
            knob_qual = ann['knob_quality'].upper()
            surface_qual = ann['surface_quality'].upper()
            
            # Helper function to classify quality
            def is_good(quality):
                return quality in ['GOOD', 'EXCELLENT', 'PERFECT']
            
            # Analyze combinations
            hole_good = is_good(hole_qual)
            text_good = is_good(text_qual)
            knob_good = is_good(knob_qual)
            surface_good = is_good(surface_qual)
            
            # Hole + Knob combination
            if hole_good and knob_good:
                combination_stats['hole_knob']['good'].append(image_name)
            else:
                combination_stats['hole_knob']['bad'].append(image_name)
            
            # Knob + Text combination
            if knob_good and text_good:
                combination_stats['knob_text']['good'].append(image_name)
            else:
                combination_stats['knob_text']['bad'].append(image_name)
            
            # Text + Surface combination
            if text_good and surface_good:
                combination_stats['text_surface']['good'].append(image_name)
            else:
                combination_stats['text_surface']['bad'].append(image_name)
            
            # Hole + Text combination
            if hole_good and text_good:
                combination_stats['hole_text']['good'].append(image_name)
            else:
                combination_stats['hole_text']['bad'].append(image_name)
            
            # Hole + Surface combination
            if hole_good and surface_good:
                combination_stats['hole_surface']['good'].append(image_name)
            else:
                combination_stats['hole_surface']['bad'].append(image_name)
            
            # Knob + Surface combination
            if knob_good and surface_good:
                combination_stats['knob_surface']['good'].append(image_name)
            else:
                combination_stats['knob_surface']['bad'].append(image_name)
            
            # All components
            if all([hole_good, text_good, knob_good, surface_good]):
                combination_stats['all_components']['good'].append(image_name)
            else:
                combination_stats['all_components']['bad'].append(image_name)
    
    return combination_stats

def create_combination_visualizations(combination_stats, output_dir=None):
    """
    Create visualizations for quality combinations.
    """
    combinations = list(combination_stats.keys())
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Quality Combination Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, combo in enumerate(combinations):
        ax = axes_flat[i]
        
        good_count = len(combination_stats[combo]['good'])
        bad_count = len(combination_stats[combo]['bad'])
        total = good_count + bad_count
        
        if total > 0:
            # Create bar plot
            labels = ['Good', 'Bad']
            values = [good_count, bad_count]
            colors = ['#2E8B57', '#DC143C']  # Green for good, red for bad
            
            bars = ax.bar(labels, values, color=colors)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Add percentage labels
            for j, (label, value) in enumerate(zip(labels, values)):
                percentage = (value / total) * 100
                ax.text(j, value/2, f'{percentage:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
            
            ax.set_title(f'{combo.replace("_", " + ").title()}', fontweight='bold')
            ax.set_ylabel('Count')
            ax.set_ylim(0, max(values) * 1.2)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{combo.replace("_", " + ").title()}')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'quality_combination_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combination visualization to: {output_path}")
    
    plt.show()

def export_csv_files(quality_stats, annotations, output_dir):
    """
    Export CSV files for individual quality categories with good vs bad samples.
    Only exports hole, knob, text, and overall quality fields.
    """
    # Only consider these quality fields (excluding surface_quality)
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'overall_quality']
    
    # Create output directory for CSV files
    csv_dir = Path(output_dir) / 'csv_exports'
    csv_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Exporting CSV files to: {csv_dir}")
    
    # Define quality categories
    good_labels = ['GOOD', 'EXCELLENT', 'PERFECT']
    bad_labels = ['BAD', 'POOR', 'DEFECTIVE', 'FAIL']
    
    # Export individual quality field CSVs
    for field in quality_fields:
        if field in quality_stats and quality_stats[field]:
            # Create DataFrame for this quality field
            data = quality_stats[field]
            
            # Separate samples by category
            good_samples = []
            bad_samples = []
            unknown_samples = []
            
            for ann in annotations:
                if field in ann:
                    quality = ann[field].upper()
                    image_name = ann.get('_image_name', 'unknown')
                    
                    if quality in good_labels:
                        good_samples.append(image_name)
                    elif quality in bad_labels:
                        bad_samples.append(image_name)
                    else:
                        unknown_samples.append(image_name)
            
            # Create distribution summary CSV
            records = []
            for quality, count in data.items():
                if quality.upper() in good_labels:
                    category = 'Good'
                elif quality.upper() in bad_labels:
                    category = 'Bad'
                else:
                    category = 'Unknown'
                
                records.append({
                    'Quality_Rating': quality,
                    'Category': category,
                    'Count': count,
                    'Percentage': (count / sum(data.values())) * 100
                })
            
            df = pd.DataFrame(records)
            csv_path = csv_dir / f'{field}_distribution.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✅ {field}_distribution.csv")
            
            # Create good samples CSV
            if good_samples:
                good_df = pd.DataFrame({
                    'Image_Name': good_samples,
                    'Quality_Field': field.replace('_', ' ').title(),
                    'Category': 'Good'
                })
                csv_path = csv_dir / f'{field}_good_samples.csv'
                good_df.to_csv(csv_path, index=False)
                print(f"  ✅ {field}_good_samples.csv ({len(good_samples)} samples)")
            
            # Create bad samples CSV
            if bad_samples:
                bad_df = pd.DataFrame({
                    'Image_Name': bad_samples,
                    'Quality_Field': field.replace('_', ' ').title(),
                    'Category': 'Bad'
                })
                csv_path = csv_dir / f'{field}_bad_samples.csv'
                bad_df.to_csv(csv_path, index=False)
                print(f"  ✅ {field}_bad_samples.csv ({len(bad_samples)} samples)")
            
            # Create unknown samples CSV
            if unknown_samples:
                unknown_df = pd.DataFrame({
                    'Image_Name': unknown_samples,
                    'Quality_Field': field.replace('_', ' ').title(),
                    'Category': 'Unknown',
                    'Quality_Label': [ann[field] for ann in annotations if field in ann and ann[field].upper() not in good_labels + bad_labels]
                })
                csv_path = csv_dir / f'{field}_unknown_samples.csv'
                unknown_df.to_csv(csv_path, index=False)
                print(f"  ✅ {field}_unknown_samples.csv ({len(unknown_samples)} samples)")
    
    # Create a summary CSV with all unknown labels found
    all_unknown_labels = set()
    for field in quality_fields:
        if field in quality_stats:
            for quality in quality_stats[field].keys():
                if quality.upper() not in good_labels + bad_labels:
                    all_unknown_labels.add(quality)
    
    if all_unknown_labels:
        unknown_summary_df = pd.DataFrame({
            'Unknown_Quality_Labels': list(all_unknown_labels),
            'Count': [sum(1 for ann in annotations if any(field in ann and ann[field] == label for field in quality_fields)) for label in all_unknown_labels]
        })
        csv_path = csv_dir / 'unknown_quality_labels_summary.csv'
        unknown_summary_df.to_csv(csv_path, index=False)
        print(f"  ✅ unknown_quality_labels_summary.csv ({len(all_unknown_labels)} unique unknown labels)")

def create_summary_combination_statistics(combination_stats):
    """
    Create a summary table of combination statistics.
    """
    print("\n" + "="*80)
    print("QUALITY COMBINATION SUMMARY STATISTICS")
    print("="*80)
    
    for combo, stats in combination_stats.items():
        good_count = len(stats['good'])
        bad_count = len(stats['bad'])
        total = good_count + bad_count
        
        if total > 0:
            good_percent = (good_count / total) * 100
            bad_percent = (bad_count / total) * 100
            
            print(f"\n{combo.replace('_', ' + ').title()}:")
            print(f"  Total samples: {total}")
            print(f"  Good quality: {good_count} ({good_percent:.1f}%)")
            print(f"  Bad quality: {bad_count} ({bad_percent:.1f}%)")
            
            if good_count > 0:
                print(f"  Good samples: {', '.join(stats['good'][:5])}{'...' if len(stats['good']) > 5 else ''}")
            if bad_count > 0:
                print(f"  Bad samples: {', '.join(stats['bad'][:5])}{'...' if len(stats['bad']) > 5 else ''}")

def main():
    """
    Main function to run the dataset analysis.
    """
    # Get the project base path
    project_base = Path(__file__).resolve().parent.parent
    
    # Default data folder path
    data_folder = project_base / 'data_v1'
    
    print("🔍 Dataset Quality Analyzer")
    print("="*50)
    print(f"Analyzing dataset in: {data_folder}")
    
    # Check if data folder exists
    if not data_folder.exists():
        print(f"❌ Data folder not found: {data_folder}")
        print("Please make sure the data folder exists and contains annotation files.")
        return
    
    # Load annotations
    annotations, annotation_files = load_annotations(data_folder)
    
    if not annotations:
        print("❌ No valid annotations found!")
        return
    
    print(f"✅ Successfully loaded {len(annotations)} annotations")
    
    # Analyze quality distributions
    quality_stats = analyze_quality_distributions(annotations)
    
    # Create summary statistics
    summary_data = create_summary_statistics(quality_stats)
    
    # Analyze defect types
    defect_stats = analyze_defect_types(annotations)
    
    # Analyze quality combinations
    print("\n🔍 Analyzing quality combinations...")
    combination_stats = analyze_quality_combinations(annotations)
    
    # Create summary combination statistics
    create_summary_combination_statistics(combination_stats)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_quality_visualizations(quality_stats, output_dir=project_base)
    create_combination_visualizations(combination_stats, output_dir=project_base)
    
    # Export CSV files
    export_csv_files(quality_stats, annotations, project_base)
    
    print("\n🎉 Analysis complete!")

if __name__ == '__main__':
    main() 