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

from model import HierarchicalQualityModel
from dataset import get_validation_augmentations
from train import BatteryQualityTrainer


class BatteryQualityInference:
    """Simple inference engine for battery quality assessment"""
    
    def __init__(self, model_path):
        """Load trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model from checkpoint
        self.model = BatteryQualityTrainer.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        # Load transforms
        self.transform = get_validation_augmentations()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"🖥️  Using device: {self.device}")
    
    def predict_single_image(self, image_path):
        """Predict quality for a single image"""
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Create dummy masks for transform (not used in inference)
        dummy_masks = np.zeros((h, w, 6), dtype=np.float32)
        
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
        
        # Add segmentation info
        seg_probs = torch.sigmoid(outputs['segmentation']).squeeze().cpu().numpy()
        predictions['segmentation'] = {
            'good_holes_pixels': int((seg_probs[0] > 0.5).sum()),
            'deformed_holes_pixels': int((seg_probs[1] > 0.5).sum()),
            'blocked_holes_pixels': int((seg_probs[2] > 0.5).sum()),
            'text_pixels': int((seg_probs[3] > 0.5).sum()),
            'plus_knob_pixels': int((seg_probs[4] > 0.5).sum()),
            'minus_knob_pixels': int((seg_probs[5] > 0.5).sum()),
        }
        
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
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Create inference engine
    inference = BatteryQualityInference(args.model_path)
    
    # Run predictions
    if args.image_path:
        # Single image
        print(f"\n🔍 Analyzing image: {args.image_path}")
        results = inference.predict_single_image(args.image_path)
        
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