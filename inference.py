import torch
import cv2
import numpy as np
from pathlib import Path

from .model import HierarchicalQualityModel
from .dataset import get_validation_augmentations

class QualityInference:
    """Inference class for the hierarchical quality model"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = HierarchicalQualityModel()
        
        # Handle different checkpoint formats
        if isinstance(checkpoint_path, (str, Path)):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle PyTorch Lightning checkpoint format
            if 'state_dict' in checkpoint:
                # Remove 'model.' prefix from keys if present
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('model.'):
                        state_dict[key[6:]] = value  # Remove 'model.' prefix
                    else:
                        state_dict[key] = value
                self.model.load_state_dict(state_dict)
            else:
                # Regular PyTorch checkpoint
                self.model.load_state_dict(checkpoint)
        else:
            # Direct model passed
            self.model = checkpoint_path
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_validation_augmentations()
    
    def predict(self, image_path, visualize=True):
        """Predict quality for a single image"""
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Assume it's already an image array
            image = image_path
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Transform
        transformed = self.transform(image=image_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Create dummy features (in real scenario, extract from image)
        dummy_features = torch.zeros(1, 12).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor, dummy_features)
        
        # Process outputs
        results = {
            'hole_quality': 'GOOD' if outputs['hole_quality'].item() > 0.5 else 'BAD',
            'hole_quality_score': outputs['hole_quality'].item(),
            'text_quality': 'GOOD' if outputs['text_quality'].item() > 0.5 else 'BAD',
            'text_quality_score': outputs['text_quality'].item(),
            'knob_quality': 'GOOD' if outputs['knob_quality'].item() > 0.5 else 'BAD',
            'knob_quality_score': outputs['knob_quality'].item(),
            'surface_quality': 'GOOD' if outputs['surface_quality'].item() > 0.5 else 'BAD',
            'surface_quality_score': outputs['surface_quality'].item(),
            'overall_quality': 'GOOD' if outputs['overall_quality'].item() > 0.5 else 'BAD',
            'overall_quality_score': outputs['overall_quality'].item(),
        }
        
        # Get segmentation masks
        seg_masks = torch.sigmoid(outputs['segmentation']).squeeze().cpu().numpy()
        
        if visualize:
            self.visualize_results(image, results, seg_masks)
        
        return results, seg_masks
    
    def predict_batch(self, image_paths, batch_size=8):
        """Predict quality for multiple images"""
        results = []
        masks = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            batch_masks = []
            
            for path in batch_paths:
                result, mask = self.predict(path, visualize=False)
                batch_results.append(result)
                batch_masks.append(mask)
            
            results.extend(batch_results)
            masks.extend(batch_masks)
        
        return results, masks
    
    def visualize_results(self, image, results, seg_masks):
        """Visualize prediction results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Segmentation masks
            mask_titles = ['Good Holes', 'Deformed Holes', 'Blocked Holes', 
                          'Text Region', 'Plus Knob', 'Minus Knob']
            
            for i, (mask, title) in enumerate(zip(seg_masks, mask_titles)):
                row = (i + 1) // 3
                col = (i + 1) % 3
                axes[row, col].imshow(mask, cmap='hot')
                axes[row, col].set_title(title)
                axes[row, col].axis('off')
            
            # Add quality scores as text
            quality_text = "Quality Assessment:\n\n"
            for component in ['hole', 'text', 'knob', 'surface', 'overall']:
                quality = results[f'{component}_quality']
                score = results[f'{component}_quality_score']
                emoji = "✅" if quality == "GOOD" else "❌"
                quality_text += f"{component.title()}: {emoji} {quality} ({score:.1%})\n"
            
            plt.figtext(0.02, 0.5, quality_text, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
            print("Quality Assessment Results:")
            for component in ['hole', 'text', 'knob', 'surface', 'overall']:
                quality = results[f'{component}_quality']
                score = results[f'{component}_quality_score']
                print(f"  {component.title()}: {quality} ({score:.1%})")

def load_model_for_inference(checkpoint_path, device='auto'):
    """Convenience function to load model for inference"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return QualityInference(checkpoint_path, device)

def predict_single_image(image_path, model_path, visualize=False):
    """Convenience function for single image prediction"""
    engine = QualityInference(model_path)
    return engine.predict(image_path, visualize=visualize) 