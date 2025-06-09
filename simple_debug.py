#!/usr/bin/env python3
"""
Simple Debug Script for Battery Quality Inspection

This script processes 1-2 samples and shows the complete pipeline:
1. Raw data loading
2. Preprocessing 
3. Model forward pass
4. Output interpretation
"""

import torch
import cv2
import numpy as np
import json
import glob
from pathlib import Path

# Import our modules
from model import HierarchicalQualityModel, ComponentAwareLoss
from dataset import ComponentQualityDataset, get_validation_augmentations

def check_gpu():
    """Check GPU availability and setup"""
    print("🖥️  GPU CHECK")
    print("-" * 30)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set device
        torch.cuda.set_device(0)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return device
    else:
        print("❌ GPU Not Available - Using CPU")
        return torch.device('cpu')

def load_single_sample():
    """Load a single sample for debugging"""
    print("\n📂 LOADING SAMPLE")
    print("-" * 30)
    
    # Find annotation files
    patterns = [
        '/o:/Amaron/extracted_frames_9182/*_enhanced_annotation.json',
        '../extracted_frames_9182/*_enhanced_annotation.json',
        'extracted_frames_9182/*_enhanced_annotation.json'
    ]
    
    annotation_file = None
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            annotation_file = files[0]  # Take first file
            print(f"✅ Found annotation: {Path(annotation_file).name}")
            break
    
    if not annotation_file:
        print("❌ No annotation files found!")
        return None
    
    # Load annotation
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)
    
    # Find corresponding image
    image_file = annotation_file.replace('_enhanced_annotation.json', '.jpg')
    
    if not Path(image_file).exists():
        print(f"❌ Image file not found: {image_file}")
        return None
    
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print(f"❌ Could not load image: {image_file}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Image loaded: {image_rgb.shape}")
    print(f"✅ Quality labels:")
    print(f"   - Hole: {annotation.get('hole_quality', 'N/A')}")
    print(f"   - Text: {annotation.get('text_quality', 'N/A')}")
    print(f"   - Knob: {annotation.get('knob_quality', 'N/A')}")
    print(f"   - Surface: {annotation.get('surface_quality', 'N/A')}")
    print(f"   - Overall: {annotation.get('overall_quality', 'N/A')}")
    
    # Debug polygon information
    print(f"✅ Polygon information:")
    print(f"   - Hole polygons: {len(annotation.get('hole_polygons', []))}")
    print(f"   - Text polygon: {'Yes' if annotation.get('text_polygon') else 'No'}")
    print(f"   - Plus knob polygon: {'Yes' if annotation.get('plus_knob_polygon') else 'No'}")
    print(f"   - Minus knob polygon: {'Yes' if annotation.get('minus_knob_polygon') else 'No'}")
    
    return {
        'image': image_rgb,
        'annotation': annotation,
        'image_file': image_file,
        'annotation_file': annotation_file
    }

def preprocess_sample(sample, device):
    """Preprocess sample through dataset pipeline"""
    print("\n🔧 PREPROCESSING")
    print("-" * 30)
    
    # Create temporary annotation file
    temp_annotations = [sample['annotation']]
    temp_file = 'temp_annotation.json'
    
    with open(temp_file, 'w') as f:
        json.dump(temp_annotations, f)
    
    try:
        # Create dataset
        transform = get_validation_augmentations()
        dataset = ComponentQualityDataset(
            annotations_file=temp_file,
            img_dir=Path(sample['image_file']).parent,
            transform=transform,
            phase='debug'
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Get processed sample
        print("   Processing sample...")
        processed = dataset[0]
        
        print("✅ Preprocessing completed")
        print(f"📊 Processed data shapes:")
        
        for key, value in processed.items():
            if torch.is_tensor(value):
                print(f"   {key}: {value.shape} ({value.dtype})")
                
                # Show mask channel explanation
                if key == 'masks':
                    print(f"      📋 Mask channels:")
                    print(f"         Channel 0: Good holes")
                    print(f"         Channel 1: Deformed holes") 
                    print(f"         Channel 2: Blocked holes")
                    print(f"         Channel 3: Text region")
                    print(f"         Channel 4: Plus knob")
                    print(f"         Channel 5: Minus knob")
                
                # Move to device
                processed[key] = value.to(device)
                
                # Show ranges for quality labels
                if 'quality' in key and value.numel() == 1:
                    print(f"      value: {value.item()}")
            else:
                print(f"   {key}: {value}")
        
        # Create batch (add batch dimension)
        batch = {}
        for key, value in processed.items():
            if torch.is_tensor(value):
                batch[key] = value.unsqueeze(0)  # Add batch dimension
            else:
                batch[key] = torch.tensor([value]).to(device)
        
        print(f"\n📦 Batch shapes:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"   {key}: {value.shape}")
        
        return processed, batch
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Cleanup
        if Path(temp_file).exists():
            Path(temp_file).unlink()

def run_model_inference(batch, device):
    """Run model inference and show outputs"""
    print("\n🤖 MODEL INFERENCE")
    print("-" * 30)
    
    # Create model with aspect ratio preserving input (960x544)
    model = HierarchicalQualityModel(input_size=(544, 960))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run inference
    with torch.no_grad():
        # Extract inputs
        images = batch['image']
        features = batch['features']
        
        print(f"\n🔍 Input shapes:")
        print(f"   Images: {images.shape}")
        print(f"   Features: {features.shape}")
        
        # Forward pass
        outputs = model(images, features)
        
        print(f"\n📤 Model outputs:")
        results = {}
        
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"   {key}: {value.shape}")
                
                if key.endswith('_quality'):
                    # Quality heads already have sigmoid, so value is already probability
                    prob = value.squeeze().cpu().item()
                    prediction = "GOOD" if prob > 0.5 else "BAD"
                    results[key] = {
                        'probability': prob,
                        'prediction': prediction,
                        'raw_output': prob  # Since it's already a probability
                    }
                    print(f"      → {prediction} ({prob:.3f})")
                elif key == 'segmentation':
                    # Show segmentation info
                    seg_probs = torch.sigmoid(value).squeeze()
                    print(f"      → Segmentation maps: {seg_probs.shape}")
                    for i, name in enumerate(['Good Holes', 'Deformed Holes', 'Blocked Holes', 'Text', 'Plus Knob', 'Minus Knob']):
                        if i < seg_probs.shape[0]:
                            active_pixels = (seg_probs[i] > 0.5).sum().item()
                            print(f"         {name}: {active_pixels} active pixels")
        
        return outputs, results

def calculate_loss(outputs, batch):
    """Calculate and show loss breakdown"""
    print("\n💰 LOSS CALCULATION")
    print("-" * 30)
    
    try:
        loss_fn = ComponentAwareLoss()
        losses = loss_fn(outputs, batch)
        
        print("✅ Loss calculation successful:")
        total_loss = 0
        
        for loss_name, loss_value in losses.items():
            print(f"   {loss_name}: {loss_value:.6f}")
            if loss_name != 'total_loss':
                total_loss += loss_value.item()
        
        print(f"\n📊 Loss breakdown:")
        print(f"   Total computed: {total_loss:.6f}")
        print(f"   Total reported: {losses['total_loss']:.6f}")
        
        return losses
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function"""
    print("🔋 SIMPLE BATTERY QUALITY DEBUG")
    print("=" * 50)
    
    # 1. Check GPU
    device = check_gpu()
    
    # 2. Load sample
    sample = load_single_sample()
    if sample is None:
        print("❌ Failed to load sample")
        return
    
    # 3. Preprocess
    processed, batch = preprocess_sample(sample, device)
    if batch is None:
        print("❌ Failed to preprocess sample")
        return
    
    # 4. Run model
    outputs, results = run_model_inference(batch, device)
    
    # 5. Calculate loss
    losses = calculate_loss(outputs, batch)
    
    # 6. Summary
    print("\n🎯 SUMMARY")
    print("=" * 50)
    print("Quality Predictions:")
    for component in ['hole', 'text', 'knob', 'surface', 'overall']:
        key = f'{component}_quality'
        if key in results:
            result = results[key]
            print(f"   {component.title()}: {result['prediction']} ({result['probability']:.3f})")
    
    if losses:
        print(f"\nTotal Loss: {losses['total_loss']:.6f}")
    
    # Memory info
    if device.type == 'cuda':
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    print("\n✅ Debug completed successfully!")
    
    return {
        'sample': sample,
        'processed': processed,
        'batch': batch,
        'outputs': outputs,
        'results': results,
        'losses': losses
    }

if __name__ == "__main__":
    main() 