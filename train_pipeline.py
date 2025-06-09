#!/usr/bin/env python3
"""
Updated Battery Quality Training Pipeline

This file now imports from the modularized components and provides
a simple interface for training and inference.
"""

# Import all components from the modularized structure
from .model import HierarchicalQualityModel, ComponentAwareLoss
from .dataset import (
    ComponentQualityDataset, 
    get_training_augmentations, 
    get_validation_augmentations
)
from .train import (
    HierarchicalQualityModule, 
    train_hierarchical_model, 
    create_submission_package
)
from .inference import QualityInference, load_model_for_inference, predict_single_image

# For backward compatibility, expose all the original classes and functions
globals().update({
    'ComponentQualityDataset': ComponentQualityDataset,
    'HierarchicalQualityModel': HierarchicalQualityModel,
    'ComponentAwareLoss': ComponentAwareLoss,
    'HierarchicalQualityModule': HierarchicalQualityModule,
    'QualityInference': QualityInference,
    'get_training_augmentations': get_training_augmentations,
    'get_validation_augmentations': get_validation_augmentations,
    'train_hierarchical_model': train_hierarchical_model,
    'create_submission_package': create_submission_package
})

def main():
    """Main training function with improved configuration"""
    
    # Example configuration for training with current annotation structure
    config = {
        # Update these paths to match your annotation files
        'train_annotations': 'path/to/train_annotations.json',  # You'll need to create this
        'train_img_dir': '../extracted_frames_9182',  # Using the available annotated frames
        'val_annotations': 'path/to/val_annotations.json',     # You'll need to create this  
        'val_img_dir': '../extracted_frames_9182',    # Using the available annotated frames
        
        # Training parameters
        'batch_size': 4,  # Reduced for limited data
        'epochs': 50,     # Reduced for initial testing
        'learning_rate': 1e-4,
        'warmup_epochs': 3,
        'num_workers': 2,
        'accumulate_grad_batches': 8  # Increased to simulate larger batch size
    }
    
    print("="*60)
    print("BATTERY QUALITY INSPECTION - TRAINING PIPELINE")
    print("="*60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Check if annotation files exist
    import os
    if not os.path.exists(config['train_annotations']):
        print("⚠️  WARNING: Training annotation file not found!")
        print(f"   Expected: {config['train_annotations']}")
        print("   Please create annotation files from the extracted frames.")
        print("   You can use the create_annotation_file() function below.")
        return
    
    try:
        # Train model
        print("🚀 Starting training...")
        model = train_hierarchical_model(config)
        print("✅ Training completed successfully!")
        
        # Create deployment package
        print("📦 Creating deployment package...")
        create_submission_package('checkpoints/best_model.ckpt', 'deployment')
        print("✅ Deployment package created!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check your configuration and annotation files.")

def create_annotation_file():
    """
    Helper function to create annotation JSON files from the existing 
    enhanced annotation files in the extracted_frames directories.
    
    This function converts individual JSON files into the format expected
    by the dataset loader.
    """
    import json
    import glob
    from pathlib import Path
    
    # Collect all annotation files
    annotation_files = []
    for pattern in ['../extracted_frames_9182/*.json', '../extracted_frames_9183/*.json', '../extracted_frames_9198/*.json']:
        annotation_files.extend(glob.glob(pattern))
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # Load all annotations
    all_annotations = []
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            all_annotations.append(ann)
        except Exception as e:
            print(f"⚠️  Error loading {ann_file}: {e}")
    
    print(f"Successfully loaded {len(all_annotations)} annotations")
    
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
    print(f"   train_annotations.json ({len(train_annotations)} samples)")
    print(f"   val_annotations.json ({len(val_annotations)} samples)")

def test_single_inference():
    """Test inference on a single image from the available dataset"""
    import glob
    
    # Find first available image
    image_files = glob.glob('../extracted_frames_9182/*.jpg')
    if not image_files:
        print("❌ No image files found for testing")
        return
    
    test_image = image_files[0]
    print(f"🧪 Testing inference on: {test_image}")
    
    # This would require a trained model checkpoint
    try:
        engine = QualityInference('checkpoints/best_model.ckpt')
        results, masks = engine.predict(test_image)
        print("✅ Inference test successful!")
        print("Results:", results)
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        print("   Make sure you have a trained model checkpoint")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "create_annotations":
            create_annotation_file()
        elif command == "test_inference":
            test_single_inference()
        elif command == "train":
            main()
        else:
            print("Usage:")
            print("  python train_pipeline.py create_annotations  # Create annotation files")
            print("  python train_pipeline.py train              # Start training")
            print("  python train_pipeline.py test_inference     # Test inference")
    else:
        print("Battery Quality Inspection Pipeline")
        print("=" * 40)
        print("Available commands:")
        print("  create_annotations - Create train/val annotation files")
        print("  train             - Start model training") 
        print("  test_inference    - Test model inference")
        print("\nExample: python train_pipeline.py create_annotations")