#!/usr/bin/env python3
"""
Example Usage of the Battery Quality Inspection System

This script demonstrates how to use the modularized components
for training and inference.
"""

import json
import glob
from pathlib import Path

# Import the modularized components
from model import HierarchicalQualityModel, ComponentAwareLoss
from dataset import ComponentQualityDataset, get_training_augmentations, get_validation_augmentations
from train import train_hierarchical_model, create_submission_package
from inference import QualityInference, predict_single_image

def create_consolidated_annotations():
    """Create consolidated annotation files from individual JSON files"""
    
    print("🔄 Creating consolidated annotation files...")
    
    # Collect all annotation files
    annotation_files = []
    for pattern in ['../extracted_frames_9182/*.json', '../extracted_frames_9183/*.json', '../extracted_frames_9198/*.json']:
        files = glob.glob(pattern)
        annotation_files.extend(files)
        print(f"   Found {len(files)} files in {pattern}")
    
    print(f"📋 Total annotation files found: {len(annotation_files)}")
    
    # Load all annotations
    all_annotations = []
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            
            # Validate annotation has required fields
            required_fields = ['image_path', 'hole_quality', 'text_quality', 'knob_quality', 'surface_quality', 'overall_quality']
            if all(field in ann for field in required_fields):
                all_annotations.append(ann)
            else:
                print(f"⚠️  Skipping {ann_file}: Missing required fields")
                
        except Exception as e:
            print(f"❌ Error loading {ann_file}: {e}")
    
    print(f"✅ Successfully loaded {len(all_annotations)} valid annotations")
    
    if len(all_annotations) == 0:
        print("❌ No valid annotations found!")
        return False
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(all_annotations))
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]
    
    # Save annotation files
    with open('train_annotations.json', 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open('val_annotations.json', 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"✅ Created annotation files:")
    print(f"   📄 train_annotations.json ({len(train_annotations)} samples)")
    print(f"   📄 val_annotations.json ({len(val_annotations)} samples)")
    
    return True

def example_training():
    """Example of how to train the model"""
    
    print("\n🚀 TRAINING EXAMPLE")
    print("=" * 50)
    
    # Create annotation files if they don't exist
    if not Path('train_annotations.json').exists():
        print("📝 Creating annotation files first...")
        if not create_consolidated_annotations():
            return
    
    # Training configuration
    config = {
        'train_annotations': 'train_annotations.json',
        'train_img_dir': '../extracted_frames_9182',
        'val_annotations': 'val_annotations.json',
        'val_img_dir': '../extracted_frames_9182',
        
        # Training parameters optimized for the available data
        'batch_size': 2,  # Small batch size for limited data
        'epochs': 20,     # Fewer epochs for quick testing
        'learning_rate': 1e-4,
        'warmup_epochs': 2,
        'num_workers': 1,  # Reduced for stability
        'accumulate_grad_batches': 16  # Simulate larger batch size
    }
    
    print("⚙️  Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        print("\n🎯 Starting model training...")
        model = train_hierarchical_model(config)
        print("✅ Training completed successfully!")
        
        # Create deployment package
        print("\n📦 Creating deployment package...")
        create_submission_package('lightning_logs/version_*/checkpoints/last.ckpt', 'deployment')
        print("✅ Deployment package created!")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def example_inference():
    """Example of how to use the model for inference"""
    
    print("\n🔍 INFERENCE EXAMPLE")
    print("=" * 50)
    
    # Find available images
    image_files = glob.glob('../extracted_frames_9182/*.jpg')
    if not image_files:
        print("❌ No image files found for inference")
        return
    
    test_image = image_files[0]
    print(f"🖼️  Testing on: {Path(test_image).name}")
    
    # Check for trained model
    import os
    checkpoint_patterns = [
        'lightning_logs/version_*/checkpoints/last.ckpt',
        'lightning_logs/version_*/checkpoints/*.ckpt',
        'checkpoints/best_model.ckpt'
    ]
    
    checkpoint_path = None
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        if checkpoints:
            checkpoint_path = checkpoints[0]
            break
    
    if checkpoint_path:
        print(f"📁 Using checkpoint: {checkpoint_path}")
        try:
            # Load model and predict
            engine = QualityInference(checkpoint_path)
            results, masks = engine.predict(test_image, visualize=False)
            
            print("✅ Inference successful!")
            print("\n📊 Results:")
            for component in ['hole', 'text', 'knob', 'surface', 'overall']:
                quality = results[f'{component}_quality']
                score = results[f'{component}_quality_score']
                emoji = "✅" if quality == "GOOD" else "❌"
                print(f"   {emoji} {component.title()}: {quality} ({score:.2%})")
                
            return results
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No trained model checkpoint found!")
        print("   Please train a model first using example_training()")

def example_dataset_analysis():
    """Analyze the dataset to understand the annotation structure"""
    
    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)
    
    # Load some annotations for analysis
    annotation_files = glob.glob('../extracted_frames_9182/*.json')[:10]
    
    print(f"🔍 Analyzing {len(annotation_files)} annotation files...")
    
    quality_counts = {
        'hole_quality': {},
        'text_quality': {},
        'knob_quality': {},
        'surface_quality': {},
        'overall_quality': {}
    }
    
    missing_fields = set()
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            
            # Count quality distributions
            for quality_type in quality_counts.keys():
                if quality_type in ann:
                    value = ann[quality_type]
                    quality_counts[quality_type][value] = quality_counts[quality_type].get(value, 0) + 1
                else:
                    missing_fields.add(quality_type)
                    
        except Exception as e:
            print(f"   ⚠️  Error reading {ann_file}: {e}")
    
    print("\n📈 Quality Distribution:")
    for quality_type, counts in quality_counts.items():
        if counts:
            print(f"\n   {quality_type}:")
            for value, count in counts.items():
                print(f"     {value}: {count}")
        else:
            print(f"\n   {quality_type}: No data found")
    
    if missing_fields:
        print(f"\n⚠️  Missing fields in some annotations: {missing_fields}")
    
    # Check annotation structure
    if annotation_files:
        with open(annotation_files[0], 'r') as f:
            sample_ann = json.load(f)
        
        print(f"\n🔍 Sample annotation structure:")
        print(f"   Available fields: {list(sample_ann.keys())}")
        
        # Check for polygons
        polygon_fields = [k for k in sample_ann.keys() if 'polygon' in k]
        print(f"   Polygon fields: {polygon_fields}")

def main():
    """Main function to demonstrate all examples"""
    
    print("🔋 BATTERY QUALITY INSPECTION SYSTEM")
    print("=" * 60)
    print("This script demonstrates the modularized battery quality inspection system")
    print("=" * 60)
    
    # Analyze dataset first
    example_dataset_analysis()
    
    # Ask user what they want to do
    print("\n" + "=" * 60)
    print("What would you like to do?")
    print("1. Train a new model")
    print("2. Test inference (requires trained model)")
    print("3. Create annotation files only")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        example_training()
    elif choice == '2':
        example_inference()
    elif choice == '3':
        create_consolidated_annotations()
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main() 