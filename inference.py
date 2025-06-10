#!/usr/bin/env python3
"""
Battery Quality Inference Script

Usage:
    python inference.py --model best_custom_maskrcnn.pth --image path/to/image.jpg
    python inference.py --model best_custom_maskrcnn.pth --data_dir data/test --save_results
"""

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

class BatteryInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Loading model on {self.device}")
        
        # Load model
        self.model = CustomMaskRCNN(num_classes=5).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Class names
        self.class_names = {0: 'background', 1: 'plus_knob', 2: 'minus_knob', 3: 'text_area', 4: 'hole'}
        self.quality_names = {0: 'GOOD', 1: 'BAD', 2: 'UNKNOWN'}
        
        # Preprocessing
        self.transform = A.Compose([
            A.Resize(height=544, width=960, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    
    def postprocess_predictions(self, outputs, orig_size, conf_threshold=0.5):
        """Convert model outputs to interpretable results"""
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
        
        # Resize masks to original size
        resized_masks = []
        for mask in detections['masks']:
            mask = (mask[0] > 0.5).astype(np.uint8)
            resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized_mask)
        detections['masks'] = np.array(resized_masks)
        
        # Custom predictions
        perspective = outputs['perspective'][0].cpu().numpy()
        perspective[0::2] *= orig_w  # x coordinates
        perspective[1::2] *= orig_h  # y coordinates
        perspective = perspective.reshape(4, 2)
        
        overall_quality = torch.argmax(outputs['overall_quality'][0]).cpu().item()
        text_color = (torch.sigmoid(outputs['text_color'][0]) > 0.5).cpu().item()
        knob_size = (torch.sigmoid(outputs['knob_size'][0]) > 0.5).cpu().item()
        
        return {
            'detections': detections,
            'perspective_points': perspective,
            'overall_quality': self.quality_names[overall_quality],
            'text_color_present': bool(text_color),
            'knob_size_good': bool(knob_size)
        }
    
    def predict_single(self, image_path, conf_threshold=0.5):
        """Run inference on a single image"""
        with torch.no_grad():
            # Preprocess
            tensor_image, orig_image, orig_size = self.preprocess_image(image_path)
            
            # Inference
            outputs = self.model([tensor_image], None)
            
            # Postprocess
            results = self.postprocess_predictions(outputs, orig_size, conf_threshold)
            
            return results, orig_image
    
    def visualize_results(self, image, results, save_path=None):
        """Visualize detection and segmentation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image with detections
        ax1.imshow(image)
        ax1.set_title('Detections & Perspective Points', fontsize=16)
        
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
        
        # Add text annotations
        info_text = f"""Quality Assessment:
Overall Quality: {results['overall_quality']}
Text Color Present: {results['text_color_present']}
Knob Size Good: {results['knob_size_good']}

Detections: {len(detections['boxes'])} objects found"""
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
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
                    
                    # Save JSON results
                    json_path = output_dir / f"{img_path.stem}_results.json"
                    with open(json_path, 'w') as f:
                        # Convert numpy arrays to lists for JSON serialization
                        json_results = results.copy()
                        json_results['detections']['boxes'] = results['detections']['boxes'].tolist()
                        json_results['detections']['scores'] = results['detections']['scores'].tolist()
                        json_results['detections']['labels'] = results['detections']['labels'].tolist()
                        json_results['perspective_points'] = results['perspective_points'].tolist()
                        del json_results['detections']['masks']  # Too large for JSON
                        
                        json.dump(json_results, f, indent=2)
                
                # Print summary
                print(f"  ✅ Quality: {results['overall_quality']}")
                print(f"     Objects: {len(results['detections']['boxes'])}")
                print(f"     Text: {results['text_color_present']}, Knob: {results['knob_size_good']}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
        
        return results_list

def main():
    parser = argparse.ArgumentParser(description='Battery Quality Inference')
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
        
        # Print results
        print("\n📊 RESULTS:")
        print(f"Overall Quality: {results['overall_quality']}")
        print(f"Text Color Present: {results['text_color_present']}")
        print(f"Knob Size Good: {results['knob_size_good']}")
        print(f"Objects Detected: {len(results['detections']['boxes'])}")
        
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
        quality_counts = {}
        total_detections = 0
        
        for result in results_list:
            quality = result['overall_quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            total_detections += len(result['detections']['boxes'])
        
        print(f"\n📈 BATCH SUMMARY:")
        print(f"Total Images: {total_images}")
        print(f"Total Detections: {total_detections}")
        print(f"Average Detections per Image: {total_detections/total_images:.1f}")
        print("Quality Distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} ({count/total_images*100:.1f}%)")

if __name__ == '__main__':
    main() 