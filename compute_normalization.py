#!/usr/bin/env python3
"""
Compute dataset-specific normalization values for battery inspection images
"""

import cv2
import numpy as np
import glob
import json
from pathlib import Path

def compute_dataset_normalization():
    """Compute mean and std for battery inspection dataset"""
    print("🔍 COMPUTING DATASET NORMALIZATION VALUES")
    print("=" * 50)
    
    # Find all annotation files
    patterns = [
        '../extracted_frames_9182/*_enhanced_annotation.json',
        'extracted_frames_9182/*_enhanced_annotation.json',
        '/o:/Amaron/extracted_frames_9182/*_enhanced_annotation.json'
    ]
    
    annotation_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            annotation_files = files
            print(f"Found {len(files)} annotation files")
            break
    
    if not annotation_files:
        print("❌ No annotation files found!")
        return None, None
    
    # Collect all pixel values
    all_pixels = []
    processed_count = 0
    
    for ann_file in annotation_files[:20]:  # Process first 20 images for speed
        # Find corresponding image
        image_path = ann_file.replace('_enhanced_annotation.json', '.jpg')
        
        if not Path(image_path).exists():
            continue
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # Convert to RGB and normalize to [0, 1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Collect pixels (reshape to get all pixels)
        pixels = image_normalized.reshape(-1, 3)  # [H*W, 3]
        all_pixels.append(pixels)
        
        processed_count += 1
        if processed_count % 5 == 0:
            print(f"   Processed {processed_count} images...")
    
    if not all_pixels:
        print("❌ No valid images found!")
        return None, None
    
    # Combine all pixels
    all_pixels = np.concatenate(all_pixels, axis=0)
    print(f"   Total pixels: {all_pixels.shape[0]:,}")
    
    # Compute statistics
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)
    
    print(f"\n📊 COMPUTED NORMALIZATION VALUES:")
    print(f"   Mean (RGB): [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"   Std  (RGB): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    
    print(f"\n🔄 COMPARISON WITH IMAGENET:")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    print(f"   ImageNet Mean: {imagenet_mean}")
    print(f"   ImageNet Std:  {imagenet_std}")
    print(f"   Mean difference: {np.abs(mean - imagenet_mean)}")
    print(f"   Std difference:  {np.abs(std - imagenet_std)}")
    
    # Save to file
    normalization_data = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'computed_from': f'{processed_count} images',
        'imagenet_mean': imagenet_mean,
        'imagenet_std': imagenet_std
    }
    
    with open('battery_normalization.json', 'w') as f:
        json.dump(normalization_data, f, indent=2)
    
    print(f"\n✅ Normalization values saved to 'battery_normalization.json'")
    
    return mean, std

if __name__ == "__main__":
    mean, std = compute_dataset_normalization()
    
    if mean is not None:
        print(f"\n🎯 USE THESE VALUES IN DATASET.PY:")
        print(f"   A.Normalize(mean={mean.tolist()}, std={std.tolist()})") 