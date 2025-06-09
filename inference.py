#!/usr/bin/env python3
"""
Battery Quality Inference

Usage:
    python inference.py --model_path best_model.ckpt --image_path test_image.jpg
    python inference.py --model_path best_model.ckpt --image_dir test_images/
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import HierarchicalQualityModel
from dataset import get_validation_augmentations
from train import BatteryQualityTrainer


class BatteryQualityInference:
    """Simple inference engine for battery quality assessment"""
    
    def __init__(self, model_path, norm_stats_file='normalization_stats.json'):
        """Load trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model from checkpoint
        self.model = BatteryQualityTrainer.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Load transforms with proper normalization
        self.transform = get_validation_augmentations(norm_stats_file)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"🖥️  Using device: {self.device}")
    
    def predict_single_image(self, image_path, save_visualization=False, output_dir=None):
        """Predict quality for a single image"""
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        
        # Create dummy masks for transform (not used in inference)
        dummy_masks = np.zeros((original_h, original_w, 6), dtype=np.float32)
        
        # Apply transforms
        transformed = self.transform(image=image_rgb, mask=dummy_masks)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Create dummy features (zeros - model will handle this)
        dummy_features = torch.zeros(1, 12, device=self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor, dummy_features)
        
        # Extract predictions
        predictions = {
            'hole_quality': self._get_quality_prediction(outputs['hole_quality']),
            'text_quality': self._get_quality_prediction(outputs['text_quality']),
            'knob_quality': self._get_quality_prediction(outputs['knob_quality']),
            'surface_quality': self._get_quality_prediction(outputs['surface_quality']),
            'overall_quality': self._get_quality_prediction(outputs['overall_quality']),
        }
        
        # Get segmentation predictions (960x544)
        seg_probs = torch.sigmoid(outputs['segmentation']).squeeze().cpu().numpy()  # Shape: (6, 544, 960)
        
        predictions['segmentation'] = {
            'good_holes_pixels': int((seg_probs[0] > 0.5).sum()),
            'deformed_holes_pixels': int((seg_probs[1] > 0.5).sum()),
            'blocked_holes_pixels': int((seg_probs[2] > 0.5).sum()),
            'text_pixels': int((seg_probs[3] > 0.5).sum()),
            'plus_knob_pixels': int((seg_probs[4] > 0.5).sum()),
            'minus_knob_pixels': int((seg_probs[5] > 0.5).sum()),
        }
        
        # Add visualization data
        predictions['visualization_data'] = {
            'original_image': image_rgb,
            'seg_masks': seg_probs,  # 6 channels, 544x960
            'original_size': (original_h, original_w),
            'model_size': (544, 960)
        }
        
        # Create visualization if requested
        if save_visualization:
            viz_path = self.create_visualization(image_path, predictions, output_dir)
            predictions['visualization_path'] = viz_path
        
        return predictions
    
    def _get_quality_prediction(self, quality_tensor):
        """Convert quality tensor to readable prediction"""
        # Apply sigmoid since model outputs logits
        prob = torch.sigmoid(quality_tensor).squeeze().cpu().item()
        quality = "GOOD" if prob > 0.5 else "BAD"
        return {
            'quality': quality,
            'confidence': float(prob),
            'raw_score': float(prob)
        }
    
    def create_visualization(self, image_path, predictions, output_dir=None):
        """Create visualization of predictions overlaid on original image"""
        
        viz_data = predictions['visualization_data']
        original_image = viz_data['original_image']
        seg_masks = viz_data['seg_masks']  # Shape: (6, 544, 960)
        original_h, original_w = viz_data['original_size']
        
        # Resize masks back to original image size
        resized_masks = []
        for channel in range(6):
            mask = seg_masks[channel]  # Shape: (544, 960)
            # Resize to original image size
            resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(resized_mask)
        
        resized_masks = np.stack(resized_masks, axis=0)  # Shape: (6, original_h, original_w)
        
        # Define colors for each mask channel
        colors = [
            [0, 255, 0],      # Channel 0: Good holes - Green
            [255, 165, 0],    # Channel 1: Deformed holes - Orange  
            [255, 0, 0],      # Channel 2: Blocked holes - Red
            [0, 255, 255],    # Channel 3: Text - Cyan
            [255, 0, 255],    # Channel 4: Plus knob - Magenta
            [255, 255, 0],    # Channel 5: Minus knob - Yellow
        ]
        
        labels = [
            "Good Holes", "Deformed Holes", "Blocked Holes", 
            "Text Region", "Plus Knob", "Minus Knob"
        ]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Individual mask channels
        for i in range(6):
            row = (i + 1) // 4
            col = (i + 1) % 4
            
            # Create colored overlay
            overlay = original_image.copy().astype(np.float32)
            mask = resized_masks[i] > 0.5  # Threshold at 0.5
            
            if mask.any():
                # Apply color where mask is positive
                for c in range(3):
                    overlay[:, :, c] = np.where(mask, 
                                              overlay[:, :, c] * 0.6 + colors[i][c] * 0.4,
                                              overlay[:, :, c])
            
            axes[row, col].imshow(overlay.astype(np.uint8))
            axes[row, col].set_title(f'{labels[i]}\n({np.sum(mask)} pixels)')
            axes[row, col].axis('off')
        
        # Combined overlay
        combined_overlay = original_image.copy().astype(np.float32)
        alpha = 0.4
        
        for i in range(6):
            mask = resized_masks[i] > 0.5
            if mask.any():
                for c in range(3):
                    combined_overlay[:, :, c] = np.where(mask,
                                                       combined_overlay[:, :, c] * (1-alpha) + colors[i][c] * alpha,
                                                       combined_overlay[:, :, c])
        
        axes[1, 3].imshow(combined_overlay.astype(np.uint8))
        
        # Add quality assessment text
        quality_text = f"""Overall: {predictions['overall_quality']['quality']} ({predictions['overall_quality']['confidence']:.2f})
Holes: {predictions['hole_quality']['quality']} ({predictions['hole_quality']['confidence']:.2f})
Text: {predictions['text_quality']['quality']} ({predictions['text_quality']['confidence']:.2f})
Knobs: {predictions['knob_quality']['quality']} ({predictions['knob_quality']['confidence']:.2f})
Surface: {predictions['surface_quality']['quality']} ({predictions['surface_quality']['confidence']:.2f})"""
        
        axes[1, 3].text(0.02, 0.98, quality_text, transform=axes[1, 3].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 3].set_title('Combined Prediction')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        if output_dir is None:
            output_dir = Path(image_path).parent / 'predictions'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        image_name = Path(image_path).stem
        viz_path = output_dir / f'{image_name}_prediction.png'
        
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Visualization saved: {viz_path}")
        return str(viz_path)
    
    def predict_directory(self, image_dir):
        """Predict quality for all images in a directory"""
        image_dir = Path(image_dir)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return {}
        
        print(f"🔍 Found {len(image_files)} images")
        
        results = {}
        for image_file in image_files:
            try:
                print(f"   Processing {image_file.name}...")
                predictions = self.predict_single_image(image_file)
                results[str(image_file)] = predictions
            except Exception as e:
                print(f"   ❌ Error processing {image_file.name}: {e}")
                results[str(image_file)] = {'error': str(e)}
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Battery Quality Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # Either single image or directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str,
                      help='Path to single image')
    group.add_argument('--image_dir', type=str,
                      help='Directory containing images')
    
    parser.add_argument('--output_path', type=str,
                       help='Path to save results JSON (optional)')
    parser.add_argument('--norm_stats', type=str, default='normalization_stats.json',
                       help='Normalization statistics file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    parser.add_argument('--viz_output_dir', type=str, default=None,
                       help='Directory to save visualizations (default: same as image dir)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Create inference engine
    inference = BatteryQualityInference(args.model_path, args.norm_stats)
    
    # Run predictions
    if args.image_path:
        # Single image
        print(f"\n🔍 Analyzing image: {args.image_path}")
        results = inference.predict_single_image(args.image_path, 
                                                save_visualization=args.visualize,
                                                output_dir=args.viz_output_dir)
        
        # Print results
        print(f"\n📊 RESULTS:")
        print(f"Overall Quality: {results['overall_quality']['quality']} "
              f"(confidence: {results['overall_quality']['confidence']:.3f})")
        
        print(f"\nComponent Quality:")
        for component in ['hole', 'text', 'knob', 'surface']:
            result = results[f'{component}_quality']
            print(f"  {component.title()}: {result['quality']} ({result['confidence']:.3f})")
        
        print(f"\nSegmentation Analysis:")
        seg = results['segmentation']
        print(f"  Good holes: {seg['good_holes_pixels']} pixels")
        print(f"  Deformed holes: {seg['deformed_holes_pixels']} pixels")
        print(f"  Blocked holes: {seg['blocked_holes_pixels']} pixels")
        print(f"  Text region: {seg['text_pixels']} pixels")
        print(f"  Plus knob: {seg['plus_knob_pixels']} pixels")
        print(f"  Minus knob: {seg['minus_knob_pixels']} pixels")
        
    else:
        # Directory
        print(f"\n🔍 Analyzing directory: {args.image_dir}")
        results = inference.predict_directory(args.image_dir)
        
        # Print summary
        total_images = len(results)
        good_count = sum(1 for r in results.values() 
                        if 'overall_quality' in r and r['overall_quality']['quality'] == 'GOOD')
        bad_count = total_images - good_count
        
        print(f"\n📊 SUMMARY:")
        print(f"Total images: {total_images}")
        print(f"GOOD quality: {good_count}")
        print(f"BAD quality: {bad_count}")
        print(f"Success rate: {good_count/total_images*100:.1f}%")
    
    # Save results if requested
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output_path}")
    
    print(f"\n✅ Inference completed!")


if __name__ == "__main__":
    main() 