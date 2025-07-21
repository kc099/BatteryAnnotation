import os
import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm

def validate_annotation(ann):
    """
    Silently validates an annotation file. Returns True if valid, False otherwise.
    Rejects any annotation with UNKNOWN quality labels.
    """
    if not ann or len(ann) == 0:
        return False
    
    # Check for at least two of the three main component types
    critical_components = ['hole_polygons', 'text_polygon', 'plus_knob_polygon']
    present_components = sum(1 for comp in critical_components if ann.get(comp) and len(ann.get(comp)) > 0)
    
    if present_components < 2:
        return False
        
    # Check that required quality labels exist and are not UNKNOWN
    quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'surface_quality', 'overall_quality']
    unknown_labels = ['UNKNOWN', 'unknown', 'Unknown']
    
    for field in quality_fields:
        if field not in ann:
            return False
        if ann[field] in unknown_labels:
            return False
        
    return True

def prepare_dataset(source_dirs, output_dir, train_ratio=0.8, seed=42):
    """
    Scans source directories, validates annotations, and splits them into
    train and validation sets in the output directory.
    """
    random.seed(seed)
    
    # Create output directories
    train_path = Path(output_dir) / 'train'
    valid_path = Path(output_dir) / 'valid'
    train_path.mkdir(parents=True, exist_ok=True)
    valid_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🧹 Clearing previous data in {output_dir}...")
    for d in [train_path, valid_path]:
        for item in d.iterdir():
            item.unlink()

    print("🔍 Finding and validating all annotations...")
    valid_files = []
    for source_dir in source_dirs:
        for ann_file in tqdm(list(Path(source_dir).glob('*_enhanced_annotation.json')), desc=f"Scanning {Path(source_dir).name}"):
            # Try to find corresponding image file with either .jpg or .png extension
            base_name = ann_file.name.replace('_enhanced_annotation.json', '')
            image_file_jpg = ann_file.with_name(base_name + '.jpg')
            image_file_png = ann_file.with_name(base_name + '.png')
            
            # Check which image file exists
            if image_file_jpg.exists():
                image_file = image_file_jpg
            elif image_file_png.exists():
                image_file = image_file_png
            else:
                print(f"⚠️  No image file found for annotation: {ann_file.name}")
                continue
            
            try:
                with open(ann_file, 'r') as f:
                    ann = json.load(f)
                if validate_annotation(ann):
                    valid_files.append((image_file, ann_file))
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"✅ Found {len(valid_files)} total valid samples.")

    # Split files
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * train_ratio)
    train_files = valid_files[:split_idx]
    valid_files = valid_files[split_idx:]
    
    print(f" splitting into {len(train_files)} training and {len(valid_files)} validation samples.")

    # Copy files to new directories
    print(" G copying training files...")
    for img, ann in tqdm(train_files, desc="Training set"):
        shutil.copy(img, train_path)
        shutil.copy(ann, train_path)
        
    print(" G copying validation files...")
    for img, ann in tqdm(valid_files, desc="Validation set"):
        shutil.copy(img, valid_path)
        shutil.copy(ann, valid_path)
        
    print("\n🎉 Dataset preparation complete!")
    print(f"   Train directory: {train_path.resolve()}")
    print(f"   Valid directory: {valid_path.resolve()}")

if __name__ == '__main__':
    
    # Base path of the project (one level up from where the script is)
    project_base = Path(__file__).resolve().parent.parent

    source_directories = [
        project_base / 'Marked',
        project_base / 'Prasad',
        project_base / 'extracted_frames_9182',
        project_base / 'extracted_frames_9183',
        project_base / 'extracted_frames_9198',
        project_base / 'extracted_frames_9213',
        project_base / 'extracted_frames_9219',
        project_base / 'extracted_frames_9215',
        
    ]
    
    output_directory = project_base / 'data_v1'
    
    prepare_dataset(source_directories, output_directory) 