#!/usr/bin/env python3
"""
Test script to analyze text color detection issues with shadowed text
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from skimage.measure import regionprops

def load_test_data(results_folder):
    """Load test data from inference results"""
    results_folder = Path(results_folder)
    
    test_cases = []
    for json_file in results_folder.glob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get corresponding image
            img_name = json_file.name.replace('_results.json', '.jpg')
            img_path = results_folder.parent / img_name
            
            if img_path.exists():
                test_cases.append({
                    'json_path': json_file,
                    'img_path': img_path,
                    'data': data
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return test_cases

def analyze_text_region(image, text_mask, case_name=""):
    """Detailed analysis of text region with multiple thresholds"""
    if text_mask.sum() == 0:
        return None
    
    # Get pixels inside the text mask
    text_pixels_mask = text_mask > 0.5
    if not np.any(text_pixels_mask):
        return None
    
    # Extract RGB values from text region
    text_pixels = image[text_pixels_mask]
    
    if len(text_pixels) == 0:
        return None
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {case_name}")
    print(f"{'='*60}")
    print(f"Total text pixels: {len(text_pixels)}")
    print(f"RGB ranges:")
    print(f"  R: {text_pixels[:, 0].min()}-{text_pixels[:, 0].max()} (mean: {text_pixels[:, 0].mean():.1f})")
    print(f"  G: {text_pixels[:, 1].min()}-{text_pixels[:, 1].max()} (mean: {text_pixels[:, 1].mean():.1f})")
    print(f"  B: {text_pixels[:, 2].min()}-{text_pixels[:, 2].max()} (mean: {text_pixels[:, 2].mean():.1f})")
    
    # Test different white thresholds
    thresholds = [150, 160, 170, 180, 190, 200, 210, 220]
    results = {}
    
    for threshold in thresholds:
        white_mask = (
            (text_pixels[:, 0] > threshold) &
            (text_pixels[:, 1] > threshold) &
            (text_pixels[:, 2] > threshold)
        )
        white_count = np.sum(white_mask)
        white_ratio = white_count / len(text_pixels)
        results[threshold] = {
            'count': white_count,
            'ratio': white_ratio
        }
        print(f"  Threshold {threshold}: {white_count} pixels ({white_ratio:.4f} ratio)")
    
    # Alternative approach: Look for bright pixels relative to background
    # Calculate brightness (luminance)
    brightness = 0.299 * text_pixels[:, 0] + 0.587 * text_pixels[:, 1] + 0.114 * text_pixels[:, 2]
    brightness_mean = np.mean(brightness)
    brightness_std = np.std(brightness)
    brightness_max = np.max(brightness)
    
    print(f"\nBrightness analysis:")
    print(f"  Mean brightness: {brightness_mean:.1f}")
    print(f"  Std brightness: {brightness_std:.1f}")
    print(f"  Max brightness: {brightness_max:.1f}")
    
    # Look for pixels brighter than mean + 1 std (adaptive threshold)
    adaptive_threshold = brightness_mean + brightness_std
    bright_pixels = brightness > adaptive_threshold
    bright_count = np.sum(bright_pixels)
    bright_ratio = bright_count / len(text_pixels)
    
    print(f"  Adaptive threshold ({adaptive_threshold:.1f}): {bright_count} pixels ({bright_ratio:.4f} ratio)")
    
    # HSV analysis - look for low saturation (white/gray text)
    text_pixels_hsv = cv2.cvtColor(text_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    
    # Low saturation indicates white/gray colors
    low_saturation_mask = text_pixels_hsv[:, 1] < 50  # Saturation < 50
    high_value_mask = text_pixels_hsv[:, 2] > 100     # Value > 100 (not too dark)
    
    white_hsv_mask = low_saturation_mask & high_value_mask
    white_hsv_count = np.sum(white_hsv_mask)
    white_hsv_ratio = white_hsv_count / len(text_pixels)
    
    print(f"  HSV white detection: {white_hsv_count} pixels ({white_hsv_ratio:.4f} ratio)")
    
    return {
        'total_pixels': len(text_pixels),
        'rgb_stats': {
            'r_mean': text_pixels[:, 0].mean(),
            'g_mean': text_pixels[:, 1].mean(),
            'b_mean': text_pixels[:, 2].mean(),
            'r_max': text_pixels[:, 0].max(),
            'g_max': text_pixels[:, 1].max(),
            'b_max': text_pixels[:, 2].max(),
        },
        'threshold_results': results,
        'brightness_stats': {
            'mean': brightness_mean,
            'std': brightness_std,
            'max': brightness_max,
            'adaptive_ratio': bright_ratio
        },
        'hsv_white_ratio': white_hsv_ratio
    }

def create_text_mask_from_bbox(image_shape, bbox):
    """Create a simple rectangular mask from bounding box"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox.astype(int)
    mask[y1:y2, x1:x2] = 1
    return mask

def test_improved_text_analysis(image, text_mask):
    """Test improved text analysis method"""
    if text_mask.sum() == 0:
        return False, 0.0, "No text mask"
    
    text_pixels_mask = text_mask > 0.5
    if not np.any(text_pixels_mask):
        return False, 0.0, "No text pixels"
    
    text_pixels = image[text_pixels_mask]
    if len(text_pixels) == 0:
        return False, 0.0, "No text pixels found"
    
    # Method 1: HSV-based white detection (better for shadows)
    text_pixels_hsv = cv2.cvtColor(text_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    
    # White/light text: low saturation + reasonable value
    low_saturation = text_pixels_hsv[:, 1] < 60   # Saturation < 60
    reasonable_value = text_pixels_hsv[:, 2] > 80  # Value > 80 (not too dark)
    
    white_hsv_mask = low_saturation & reasonable_value
    white_hsv_ratio = np.sum(white_hsv_mask) / len(text_pixels)
    
    # Method 2: Adaptive brightness threshold
    brightness = 0.299 * text_pixels[:, 0] + 0.587 * text_pixels[:, 1] + 0.114 * text_pixels[:, 2]
    brightness_mean = np.mean(brightness)
    brightness_std = np.std(brightness)
    
    # Look for pixels significantly brighter than average
    adaptive_threshold = max(brightness_mean + 0.5 * brightness_std, 120)  # At least 120
    bright_mask = brightness > adaptive_threshold
    bright_ratio = np.sum(bright_mask) / len(text_pixels)
    
    # Method 3: Relaxed RGB threshold for shadowed areas
    relaxed_threshold = 160  # Lower than original 200
    relaxed_white_mask = (
        (text_pixels[:, 0] > relaxed_threshold) &
        (text_pixels[:, 1] > relaxed_threshold) &
        (text_pixels[:, 2] > relaxed_threshold)
    )
    relaxed_white_ratio = np.sum(relaxed_white_mask) / len(text_pixels)
    
    # Combined decision: any method indicates white text
    min_ratio_threshold = 0.008  # 0.8% instead of 1%
    
    hsv_sufficient = white_hsv_ratio >= min_ratio_threshold
    bright_sufficient = bright_ratio >= min_ratio_threshold
    relaxed_sufficient = relaxed_white_ratio >= min_ratio_threshold
    
    # Contrast check (still important)
    contrast_sufficient = brightness_std >= 15  # Slightly lower threshold
    
    # Decision: (any white detection method) AND contrast
    white_detected = hsv_sufficient or bright_sufficient or relaxed_sufficient
    is_good = white_detected and contrast_sufficient
    
    status = f"HSV: {white_hsv_ratio:.3f} ({'✓' if hsv_sufficient else '✗'}), "
    status += f"Bright: {bright_ratio:.3f} ({'✓' if bright_sufficient else '✗'}), "
    status += f"Relaxed: {relaxed_white_ratio:.3f} ({'✓' if relaxed_sufficient else '✗'}), "
    status += f"Contrast: {brightness_std:.1f} ({'✓' if contrast_sufficient else '✗'})"
    
    return is_good, max(white_hsv_ratio, bright_ratio, relaxed_white_ratio), status

def main():
    # Test with the problematic folder
    results_folder = "../extracted_frames_9182/inference_results"
    
    if not Path(results_folder).exists():
        print(f"Folder {results_folder} not found. Please check the path.")
        return
    
    test_cases = load_test_data(results_folder)
    print(f"Loaded {len(test_cases)} test cases")
    
    # Analyze cases with different white ratios
    good_cases = []
    bad_cases = []
    
    for case in test_cases[:10]:  # Analyze first 10 cases
        data = case['data']
        
        # Load image
        image = cv2.imread(str(case['img_path']))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get text analysis results
        text_analysis = data['analysis_details']['text_analysis']
        current_white_ratio = text_analysis['white_score']
        current_good = text_analysis['good']
        
        # Create text mask from detections (simplified)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        detections = data['detections']
        
        for box, label in zip(detections['boxes'], detections['labels']):
            if label == 3:  # text_area
                text_mask = create_text_mask_from_bbox(image.shape, np.array(box))
                break
        
        if text_mask.sum() == 0:
            continue
        
        case_name = f"{case['img_path'].name} (Current: {current_white_ratio:.3f}, Good: {current_good})"
        
        # Detailed analysis
        analysis_result = analyze_text_region(image, text_mask, case_name)
        
        if analysis_result:
            # Test improved method
            improved_good, improved_score, improved_status = test_improved_text_analysis(image, text_mask)
            
            print(f"\nCURRENT METHOD: Good={current_good}, Score={current_white_ratio:.3f}")
            print(f"IMPROVED METHOD: Good={improved_good}, Score={improved_score:.3f}")
            print(f"IMPROVED STATUS: {improved_status}")
            
            if current_white_ratio == 0.0 and improved_score > 0.0:
                print("🔥 IMPROVEMENT DETECTED: Found white text where current method failed!")
            
            print("-" * 60)
        
        if current_good:
            good_cases.append(case)
        else:
            bad_cases.append(case)
    
    print(f"\n📊 SUMMARY:")
    print(f"Good cases: {len(good_cases)}")
    print(f"Bad cases: {len(bad_cases)}")

if __name__ == "__main__":
    main() 