#!/usr/bin/env python3
"""
Debug Pipeline for Battery Quality Inspection

This script shows data transformation at each step and model outputs
for debugging and understanding the pipeline.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import our modules
from model import HierarchicalQualityModel, ComponentAwareLoss
from dataset import ComponentQualityDataset, get_training_augmentations, get_validation_augmentations
# from inference import QualityInference  # Skip for now to avoid import issues

def print_gpu_info():
    """Print detailed GPU information"""
    print("🖥️  GPU INFORMATION")
    print("=" * 50)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    else:
        print("❌ CUDA not available - will use CPU")
    print("=" * 50)

def load_sample_data(num_samples=2):
    """Load sample annotations and images"""
    print(f"\n📂 LOADING {num_samples} SAMPLE(S)")
    print("=" * 50)
    
    # Find annotation files - check multiple possible locations
    annotation_files = []
    possible_paths = [
        '../extracted_frames_9182/*_enhanced_annotation.json',
        'extracted_frames_9182/*_enhanced_annotation.json',
        '/o:/Amaron/extracted_frames_9182/*_enhanced_annotation.json',
        '../../extracted_frames_9182/*_enhanced_annotation.json'
    ]
    
    for path_pattern in possible_paths:
        files = glob.glob(path_pattern)
        if files:
            annotation_files = files
            print(f"   Found annotation files in: {path_pattern}")
            break
    
    if not annotation_files:
        print(f"❌ No annotation files found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return []
    
    if len(annotation_files) < num_samples:
        print(f"⚠️  Only found {len(annotation_files)} annotation files")
        num_samples = len(annotation_files)
    
    samples = []
    for i in range(num_samples):
        ann_file = annotation_files[i]
        print(f"\n📄 Loading: {Path(ann_file).name}")
        
        # Load annotation
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        # Find corresponding image
        image_path = ann_file.replace('_enhanced_annotation.json', '.jpg')
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            continue
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"   ✅ Image shape: {image_rgb.shape}")
        print(f"   ✅ Annotation keys: {list(annotation.keys())}")
        print(f"   ✅ Quality labels: hole={annotation.get('hole_quality')}, "
              f"text={annotation.get('text_quality')}, "
              f"knob={annotation.get('knob_quality')}, "
              f"overall={annotation.get('overall_quality')}")
        
        samples.append({
            'image_path': image_path,
            'image': image_rgb,
            'annotation': annotation
        })
    
    print(f"\n✅ Successfully loaded {len(samples)} samples")
    return samples

def debug_dataset_preprocessing(samples):
    """Debug dataset preprocessing step by step"""
    print(f"\n🔧 DATASET PREPROCESSING DEBUG")
    print("=" * 50)
    
    # Create a temporary dataset
    temp_ann_file = 'temp_debug_annotations.json'
    annotations = [sample['annotation'] for sample in samples]
    
    with open(temp_ann_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Create dataset instances
    train_transform = get_training_augmentations()
    val_transform = get_validation_augmentations()
    
    print("\n🎯 Training Augmentations:")
    print(train_transform)
    
    print("\n🎯 Validation Augmentations:")
    print(val_transform)
    
    # Test dataset loading
    try:
        dataset = ComponentQualityDataset(
            annotations_file=temp_ann_file,
            img_dir=Path(samples[0]['image_path']).parent,
            transform=val_transform,
            phase='debug'
        )
        
        print(f"\n✅ Dataset created successfully")
        print(f"   Dataset length: {len(dataset)}")
        
        # Get first sample
        sample_data = dataset[0]
        
        print(f"\n📊 SAMPLE DATA STRUCTURE:")
        for key, value in sample_data.items():
            if torch.is_tensor(value):
                print(f"   {key}: tensor shape {value.shape}, dtype {value.dtype}")
                if key != 'image':  # Don't print full image tensor
                    print(f"      value range: [{value.min():.3f}, {value.max():.3f}]")
                    if value.numel() <= 20:  # Only print small tensors
                        print(f"      values: {value}")
            else:
                print(f"   {key}: {type(value)} = {value}")
        
        # Analyze masks
        masks = sample_data['masks']
        print(f"\n🎭 MASK ANALYSIS:")
        mask_names = ['Good Holes', 'Deformed Holes', 'Blocked Holes', 'Text', 'Plus Knob', 'Minus Knob']
        for i, name in enumerate(mask_names):
            mask_sum = masks[i].sum().item()
            print(f"   {name}: {mask_sum} pixels")
        
        return sample_data, dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Cleanup
        if Path(temp_ann_file).exists():
            Path(temp_ann_file).unlink()

def debug_model_architecture():
    """Debug model architecture and initialization"""
    print(f"\n🏗️  MODEL ARCHITECTURE DEBUG")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with correct input size
    model = HierarchicalQualityModel(input_size=(544, 960))
    model.to(device)
    
    print(f"\n📐 MODEL SUMMARY:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print model structure
    print(f"\n🏛️  MODEL STRUCTURE:")
    for name, module in model.named_children():
        print(f"   {name}: {module.__class__.__name__}")
        if hasattr(module, '__len__'):
            try:
                print(f"      layers: {len(module)}")
            except:
                pass
    
    # Test forward pass with dummy data
    print(f"\n🧪 FORWARD PASS TEST:")
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 544, 960).to(device)  # Match our input size
    dummy_features = torch.randn(batch_size, 12).to(device)
    
    print(f"   Input image shape: {dummy_image.shape}")
    print(f"   Input features shape: {dummy_features.shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_image, dummy_features)
    
    print(f"\n📤 MODEL OUTPUTS:")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
            print(f"      range: [{value.min():.4f}, {value.max():.4f}]")
            if key.endswith('_quality'):
                print(f"      probabilities: {value.squeeze().cpu().numpy()}")
        else:
            print(f"   {key}: {type(value)}")
    
    return model, outputs

def debug_loss_calculation(model_outputs, sample_data):
    """Debug loss calculation"""
    print(f"\n💸 LOSS CALCULATION DEBUG")
    print("=" * 50)
    
    # Get batch size from model outputs
    batch_size = model_outputs['segmentation'].shape[0]
    
    # Create batch from sample with matching batch size
    device = next(iter(model_outputs.values())).device  # Get device from model outputs
    batch = {}
    for key, value in sample_data.items():
        if torch.is_tensor(value):
            # Repeat sample to match model output batch size
            repeated_value = value.unsqueeze(0).repeat(batch_size, *([1] * value.ndim))
            batch[key] = repeated_value.to(device)
        else:
            batch[key] = torch.tensor([value] * batch_size).to(device)
    
    # Create loss function
    loss_fn = ComponentAwareLoss()
    
    print(f"📊 BATCH DATA FOR LOSS:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"   {key}: shape {value.shape}")
    
    # Calculate losses
    try:
        losses = loss_fn(model_outputs, batch)
        
        print(f"\n💰 LOSS BREAKDOWN:")
        for loss_name, loss_value in losses.items():
            print(f"   {loss_name}: {loss_value:.6f}")
        
        return losses
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_complete_pipeline(num_samples=2):
    """Run complete pipeline debug"""
    print("🔋 BATTERY QUALITY INSPECTION - COMPLETE PIPELINE DEBUG")
    print("=" * 80)
    
    # 1. GPU Info
    print_gpu_info()
    
    # 2. Load sample data
    samples = load_sample_data(num_samples)
    if not samples:
        print("❌ No samples loaded, exiting")
        return
    
    # 3. Debug dataset preprocessing
    sample_data, dataset = debug_dataset_preprocessing(samples)
    if sample_data is None:
        print("❌ Dataset preprocessing failed, exiting")
        return
    
    # 4. Debug model architecture
    model, model_outputs = debug_model_architecture()
    
    # 5. Debug loss calculation
    losses = debug_loss_calculation(model_outputs, sample_data)
    
    # 6. Memory usage
    if torch.cuda.is_available():
        print(f"\n💾 MEMORY USAGE AFTER PIPELINE:")
        print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"   GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    
    print(f"\n✅ PIPELINE DEBUG COMPLETED SUCCESSFULLY!")
    return samples, sample_data, model, model_outputs, losses

def visualize_sample_data(samples, sample_data):
    """Visualize the loaded sample data"""
    print(f"\n📊 VISUALIZING SAMPLE DATA")
    print("=" * 50)
    
    if not samples:
        print("❌ No samples to visualize")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(samples[0]['image'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Preprocessed image (denormalized)
    img_tensor = sample_data['image']
    # Denormalize with battery-specific values
    mean = torch.tensor([0.4045, 0.4045, 0.4045])
    std = torch.tensor([0.2256, 0.2254, 0.2782])
    denorm_img = img_tensor * std[:, None, None] + mean[:, None, None]
    denorm_img = torch.clamp(denorm_img, 0, 1)
    
    axes[1].imshow(denorm_img.permute(1, 2, 0))
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')
    
    # Masks
    masks = sample_data['masks']
    mask_names = ['Good Holes', 'Deformed Holes', 'Blocked Holes', 'Text', 'Plus Knob', 'Minus Knob']
    
    for i, (mask, name) in enumerate(zip(masks, mask_names)):
        if i + 2 < len(axes):
            axes[i + 2].imshow(mask, cmap='hot')
            axes[i + 2].set_title(f'{name}\n({mask.sum():.0f} pixels)')
            axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'debug_visualization.png'")
    plt.show()

def main():
    """Main debug function"""
    print("Starting comprehensive pipeline debug...")
    
    # Run complete pipeline debug
    results = debug_complete_pipeline(num_samples=2)
    
    if results:
        samples, sample_data, model, model_outputs, losses = results
        
        # Create visualization
        try:
            visualize_sample_data(samples, sample_data)
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
        
        print(f"\n🎉 DEBUG SESSION COMPLETED!")
        print(f"   Check 'debug_visualization.png' for visual results")
        
        return results
    else:
        print(f"❌ Debug session failed")
        return None

if __name__ == "__main__":
    main() 