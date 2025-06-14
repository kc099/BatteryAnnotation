#!/usr/bin/env python3
"""
Battery Quality Inference Script

Usage:
    python inference.py --model best_custom_maskrcnn.pth --image path/to/image.jpg
    python inference.py --model best_custom_maskrcnn.pth --data_dir data/test --save_results
"""

import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from custom_maskrcnn import CustomMaskRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage
from skimage.measure import regionprops

class BatteryInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Loading model on {self.device}")
        
        # Load model
        self.model = CustomMaskRCNN(num_classes=5).to(self.device)
        try:
            # Try loading with weights_only=True first (safer)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except:
            # Fallback to weights_only=False for older model files
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class names
        self.class_names = {0: 'background', 1: 'plus_knob', 2: 'minus_knob', 3: 'text_area', 4: 'hole'}
        
        # Preprocessing - removed ImageNet normalization for better field robustness
        self.transform = A.Compose([
            A.Resize(height=544, width=960, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Just convert to [0,1] range
            ToTensorV2()
        ])
        
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor_image = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor_image, image, (orig_h, orig_w)
    
    def calculate_mask_eccentricity(self, mask):
        """Calculate eccentricity of a binary mask"""
        if mask.sum() == 0:
            return 0.0
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse if we have enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            # ellipse = ((center_x, center_y), (width, height), angle)
            width, height = ellipse[1]
            
            # Calculate eccentricity: e = sqrt(1 - (b²/a²)) where a > b
            a = max(width, height) / 2  # semi-major axis
            b = min(width, height) / 2  # semi-minor axis
            
            if a == 0:
                return 0.0
            
            eccentricity = np.sqrt(1 - (b**2 / a**2))
            return eccentricity
        
        return 0.0
    
    def analyze_hole_quality(self, hole_mask):
        """Analyze hole quality based on eccentricity"""
        if hole_mask.sum() == 0:
            return False, 0.0, "No hole detected"
        
        eccentricity = self.calculate_mask_eccentricity(hole_mask)
        is_good = eccentricity < 0.4
        
        status = f"Eccentricity: {eccentricity:.3f} ({'GOOD' if is_good else 'BAD'} - threshold: 0.4)"
        return is_good, eccentricity, status
    
    def analyze_knob_sizes(self, plus_mask, minus_mask, hole_mask):
        """Analyze knob sizes with spatial awareness"""
        if plus_mask.sum() == 0 or minus_mask.sum() == 0:
            return False, 0.0, "Missing knob masks"
        
        if hole_mask.sum() == 0:
            return False, 0.0, "No hole detected for spatial reference"
        
        # Calculate centroids
        plus_props = regionprops(plus_mask.astype(int))
        minus_props = regionprops(minus_mask.astype(int))
        hole_props = regionprops(hole_mask.astype(int))
        
        if not plus_props or not minus_props or not hole_props:
            return False, 0.0, "Could not calculate centroids"
        
        plus_centroid = plus_props[0].centroid
        minus_centroid = minus_props[0].centroid
        hole_centroid = hole_props[0].centroid
        
        # Check spatial constraint: minus knob should be closer to hole
        plus_to_hole_dist = np.sqrt((plus_centroid[0] - hole_centroid[0])**2 + 
                                   (plus_centroid[1] - hole_centroid[1])**2)
        minus_to_hole_dist = np.sqrt((minus_centroid[0] - hole_centroid[0])**2 + 
                                    (minus_centroid[1] - hole_centroid[1])**2)
        
        spatial_constraint_ok = minus_to_hole_dist < plus_to_hole_dist
        
        # Calculate area ratio
        plus_area = plus_mask.sum()
        minus_area = minus_mask.sum()
        area_ratio = plus_area / minus_area if minus_area > 0 else 0
        
        # Both constraints must be met
        size_constraint_ok = area_ratio > 1.2
        is_good = spatial_constraint_ok and size_constraint_ok
        
        status = f"Area ratio: {area_ratio:.3f} ({'GOOD' if size_constraint_ok else 'BAD'} - threshold: 1.2), "
        status += f"Spatial: {'GOOD' if spatial_constraint_ok else 'BAD'} (minus closer to hole)"
        
        return is_good, area_ratio, status
    
    def analyze_text_color(self, image, text_mask):
        """Analyze text color by examining RGB pixel distributions in masked region - improved for shadows"""
        if text_mask.sum() == 0:
            return False, 0.0, "No text mask detected"
        
        # Get pixels inside the text mask
        text_pixels_mask = text_mask > 0.5
        if not np.any(text_pixels_mask):
            return False, 0.0, "No text pixels in mask"
        
        # Extract RGB values from text region (from original RGB image)
        text_pixels = image[text_pixels_mask]  # [N, 3] where N is number of text pixels
        
        if len(text_pixels) == 0:
            return False, 0.0, "No text pixels found"
        
        # Method 1: HSV-based white detection (better for shadows)
        text_pixels_hsv = cv2.cvtColor(text_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        # White/light text: low saturation + reasonable value (works in shadows)
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
        
        # Improved decision logic: require stronger evidence for good text
        min_ratio_threshold = 0.008  # 0.8% instead of 1%
        
        hsv_sufficient = white_hsv_ratio >= min_ratio_threshold
        bright_sufficient = bright_ratio >= min_ratio_threshold
        relaxed_sufficient = relaxed_white_ratio >= min_ratio_threshold
        
        # Contrast check (still important)
        contrast_sufficient = brightness_std >= 18  # Slightly higher threshold
        
        # Improved decision logic:
        # 1. HSV method is most reliable for shadows - if it works well, trust it
        # 2. Otherwise, require at least 2 methods to agree OR very high scores
        
        hsv_strong = white_hsv_ratio >= 0.15  # Strong HSV detection (15%+)
        bright_strong = bright_ratio >= 0.20   # Strong brightness detection (20%+)
        
        # Good text requires:
        # Option 1: Strong HSV detection (works in shadows) + contrast
        # Option 2: At least 2 methods agree + contrast  
        # Option 3: Very strong brightness detection + contrast
        
        methods_agreeing = sum([hsv_sufficient, bright_sufficient, relaxed_sufficient])
        
        condition1 = hsv_strong and contrast_sufficient  # Strong HSV + contrast
        condition2 = (methods_agreeing >= 2) and contrast_sufficient  # 2+ methods + contrast
        condition3 = bright_strong and contrast_sufficient  # Very strong brightness + contrast
        
        is_good = condition1 or condition2 or condition3
        
        # Return the best score from all methods
        best_score = max(white_hsv_ratio, bright_ratio, relaxed_white_ratio)
        
        # Create detailed status with decision reasoning
        status = f"HSV: {white_hsv_ratio:.3f} ({'✓' if hsv_sufficient else '✗'}), "
        status += f"Bright: {bright_ratio:.3f} ({'✓' if bright_sufficient else '✗'}), "
        status += f"Relaxed: {relaxed_white_ratio:.3f} ({'✓' if relaxed_sufficient else '✗'}), "
        status += f"Contrast: {brightness_std:.1f} ({'✓' if contrast_sufficient else '✗'})"
        
        # Add decision reasoning
        if is_good:
            if condition1:
                status += " [Strong HSV]"
            elif condition2:
                status += f" [{methods_agreeing} methods agree]"
            elif condition3:
                status += " [Strong brightness]"
        else:
            status += " [Insufficient evidence]"
        
        return is_good, best_score, status
    
    def postprocess_predictions(self, outputs, orig_image, orig_size, conf_threshold=0.5):
        """Convert model outputs to interpretable results using mask-based analysis"""
        orig_h, orig_w = orig_size
        
        # MaskRCNN outputs
        maskrcnn_out = outputs['maskrcnn'][0]
        
        # Filter by confidence
        keep = maskrcnn_out['scores'] > conf_threshold
        
        detections = {
            'boxes': maskrcnn_out['boxes'][keep].cpu().numpy(),
            'scores': maskrcnn_out['scores'][keep].cpu().numpy(),
            'labels': maskrcnn_out['labels'][keep].cpu().numpy(),
            'masks': maskrcnn_out['masks'][keep].cpu().numpy()
        }
        
        # Scale boxes back to original image size
        scale_x = orig_w / 960
        scale_y = orig_h / 544
        detections['boxes'][:, [0, 2]] *= scale_x
        detections['boxes'][:, [1, 3]] *= scale_y
        
        # Resize masks to original size and separate by class
        plus_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        minus_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        text_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        hole_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        resized_masks = []
        for i, (mask, label) in enumerate(zip(detections['masks'], detections['labels'])):
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            resized_mask = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized_mask)
            
            # Accumulate masks by class
            if label == 1:  # plus_knob
                plus_mask = np.maximum(plus_mask, resized_mask)
            elif label == 2:  # minus_knob
                minus_mask = np.maximum(minus_mask, resized_mask)
            elif label == 3:  # text_area
                text_mask = np.maximum(text_mask, resized_mask)
            elif label == 4:  # hole
                hole_mask = np.maximum(hole_mask, resized_mask)
        
        detections['masks'] = np.array(resized_masks)
        
        # Perspective points (keep original implementation)
        perspective = outputs['perspective'][0].cpu().numpy()
        perspective[0::2] *= orig_w  # x coordinates
        perspective[1::2] *= orig_h  # y coordinates
        perspective = perspective.reshape(4, 2)
        
        # === MASK-BASED QUALITY ANALYSIS ===
        
        # 1. Hole quality analysis
        hole_good, hole_eccentricity, hole_status = self.analyze_hole_quality(hole_mask)
        
        # 2. Knob size analysis
        knob_good, knob_ratio, knob_status = self.analyze_knob_sizes(plus_mask, minus_mask, hole_mask)
        
        # 3. Text color analysis
        text_good, text_score, text_status = self.analyze_text_color(orig_image, text_mask)
        
        # 4. Overall quality (rule-based)
        overall_good = hole_good and knob_good and text_good
        overall_quality = "GOOD" if overall_good else "BAD"
        
        # Detailed analysis results
        analysis_details = {
            'hole_analysis': {
                'good': hole_good,
                'eccentricity': hole_eccentricity,
                'status': hole_status
            },
            'knob_analysis': {
                'good': knob_good,
                'area_ratio': knob_ratio,
                'status': knob_status
            },
            'text_analysis': {
                'good': text_good,
                'white_score': text_score,
                'status': text_status
            }
        }
        
        return {
            'detections': detections,
            'perspective_points': perspective,
            'overall_quality': overall_quality,
            'hole_good': hole_good,
            'text_color_good': text_good,
            'knob_size_good': knob_good,
            'analysis_details': analysis_details
        }
    
    def predict_single(self, image_path, conf_threshold=0.5):
        """Run inference on a single image"""
        with torch.no_grad():
            # Preprocess
            tensor_image, orig_image, orig_size = self.preprocess_image(image_path)
            
            # Remove batch dimension for Mask R-CNN - it expects list of [C, H, W] tensors
            tensor_image = tensor_image.squeeze(0)  # [1, 3, H, W] -> [3, H, W]
            
            # Inference
            outputs = self.model([tensor_image], None)
            
            # Postprocess with mask-based analysis
            results = self.postprocess_predictions(outputs, orig_image, orig_size, conf_threshold)
            
            return results, orig_image
    
    def visualize_results(self, image, results, save_path=None):
        """Visualize detection and segmentation results with detailed analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Original image with detections
        ax1.imshow(image)
        ax1.set_title('Detections & Analysis', fontsize=16)
        
        # Draw bounding boxes and labels
        detections = results['detections']
        for i, (box, score, label) in enumerate(zip(detections['boxes'], detections['scores'], detections['labels'])):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                   edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f"{self.class_names[label]}: {score:.2f}", 
                    fontsize=10, color='red', weight='bold')
        
        # Draw perspective points
        persp = results['perspective_points']
        if np.any(persp > 0):
            for i, (x, y) in enumerate(persp):
                ax1.plot(x, y, 'go', markersize=8)
                ax1.text(x+5, y+5, f'P{i+1}', fontsize=10, color='green', weight='bold')
        
        # Image with masks
        ax2.imshow(image)
        ax2.set_title('Segmentation Masks', fontsize=16)
        
        # Overlay masks with different colors
        colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5)]
        for i, (mask, label) in enumerate(zip(detections['masks'], detections['labels'])):
            if mask.sum() > 0:
                color = colors[i % len(colors)]
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0] = color
                ax2.imshow(colored_mask)
        
        # Add detailed analysis text
        analysis = results['analysis_details']
        info_text = f"""🔍 MASK-BASED QUALITY ANALYSIS:

Overall Quality: {results['overall_quality']}

🕳️ HOLE ANALYSIS:
{analysis['hole_analysis']['status']}

🔘 KNOB ANALYSIS:
{analysis['knob_analysis']['status']}

📝 TEXT ANALYSIS:
{analysis['text_analysis']['status']}

📊 DETECTIONS: {len(detections['boxes'])} objects found"""
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        for ax in [ax1, ax2]:
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_batch(self, data_dir, save_results=False, conf_threshold=0.5):
        """Run inference on a directory of images"""
        data_dir = Path(data_dir)
        
        # Find all image files
        image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        print(f"🔍 Found {len(image_files)} images in {data_dir}")
        
        results_list = []
        output_dir = data_dir / "inference_results" if save_results else None
        
        if save_results:
            output_dir.mkdir(exist_ok=True)
            print(f"💾 Results will be saved to {output_dir}")
        
        for img_path in image_files:
            print(f"🔮 Processing: {img_path.name}")
            
            try:
                results, orig_image = self.predict_single(img_path, conf_threshold)
                
                # Add metadata
                results['image_path'] = str(img_path)
                results['image_name'] = img_path.name
                results_list.append(results)
                
                if save_results:
                    # Save visualization
                    vis_path = output_dir / f"{img_path.stem}_inference.png"
                    self.visualize_results(orig_image, results, vis_path)
                    
                    # Save detailed JSON results
                    json_path = output_dir / f"{img_path.stem}_results.json"
                    with open(json_path, 'w') as f:
                        # Convert numpy arrays to lists for JSON serialization
                        json_results = results.copy()
                        json_results['detections']['boxes'] = results['detections']['boxes'].tolist()
                        json_results['detections']['scores'] = results['detections']['scores'].tolist()
                        json_results['detections']['labels'] = results['detections']['labels'].tolist()
                        json_results['perspective_points'] = results['perspective_points'].tolist()
                        del json_results['detections']['masks']  # Too large for JSON
                        
                        # Convert numpy booleans to Python booleans for JSON serialization
                        json_results['hole_good'] = bool(results['hole_good'])
                        json_results['text_color_good'] = bool(results['text_color_good'])
                        json_results['knob_size_good'] = bool(results['knob_size_good'])
                        
                        # Convert nested analysis details
                        for analysis_type in ['hole_analysis', 'knob_analysis', 'text_analysis']:
                            if 'good' in json_results['analysis_details'][analysis_type]:
                                json_results['analysis_details'][analysis_type]['good'] = bool(
                                    json_results['analysis_details'][analysis_type]['good']
                                )
                        
                        json.dump(json_results, f, indent=2)
                
                # Print summary
                print(f"  ✅ Quality: {results['overall_quality']}")
                print(f"     Hole: {results['hole_good']}, Text: {results['text_color_good']}, Knob: {results['knob_size_good']}")
                print(f"     Objects: {len(results['detections']['boxes'])}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results_list

def main():
    parser = argparse.ArgumentParser(description='Battery Quality Inference with Mask-Based Analysis')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Single image to process')
    parser.add_argument('--data_dir', type=str, help='Directory of images to process')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save_results', action='store_true', help='Save visualization and JSON results')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    if not args.image and not args.data_dir:
        parser.error("Must specify either --image or --data_dir")
    
    # Initialize inference
    inference = BatteryInference(args.model)
    
    if args.image:
        # Single image inference
        print(f"🔮 Processing single image: {args.image}")
        results, orig_image = inference.predict_single(args.image, args.conf_threshold)
        
        # Print detailed results
        print("\n📊 DETAILED RESULTS:")
        print(f"Overall Quality: {results['overall_quality']}")
        print(f"\n🕳️ Hole Analysis: {results['analysis_details']['hole_analysis']['status']}")
        print(f"🔘 Knob Analysis: {results['analysis_details']['knob_analysis']['status']}")
        print(f"📝 Text Analysis: {results['analysis_details']['text_analysis']['status']}")
        print(f"\n📦 Objects Detected: {len(results['detections']['boxes'])}")
        
        for i, (score, label) in enumerate(zip(results['detections']['scores'], results['detections']['labels'])):
            print(f"  {i+1}. {inference.class_names[label]}: {score:.3f}")
        
        # Visualize
        save_path = args.output_dir if args.save_results and args.output_dir else None
        inference.visualize_results(orig_image, results, save_path)
    
    elif args.data_dir:
        # Batch inference
        results_list = inference.predict_batch(args.data_dir, args.save_results, args.conf_threshold)
        
        # Summary statistics
        total_images = len(results_list)
        quality_counts = {'GOOD': 0, 'BAD': 0}
        hole_good_count = 0
        text_good_count = 0
        knob_good_count = 0
        total_detections = 0
        
        for result in results_list:
            quality_counts[result['overall_quality']] += 1
            if result['hole_good']:
                hole_good_count += 1
            if result['text_color_good']:
                text_good_count += 1
            if result['knob_size_good']:
                knob_good_count += 1
            total_detections += len(result['detections']['boxes'])
        
        print(f"\n📈 BATCH SUMMARY:")
        print(f"Total Images: {total_images}")
        print(f"Total Detections: {total_detections}")
        print(f"Average Detections per Image: {total_detections/total_images:.1f}")
        print("\n🎯 Quality Distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} ({count/total_images*100:.1f}%)")
        print("\n🔍 Component Analysis:")
        print(f"  Hole Good: {hole_good_count} ({hole_good_count/total_images*100:.1f}%)")
        print(f"  Text Good: {text_good_count} ({text_good_count/total_images*100:.1f}%)")
        print(f"  Knob Good: {knob_good_count} ({knob_good_count/total_images*100:.1f}%)")

if __name__ == '__main__':
    main() 