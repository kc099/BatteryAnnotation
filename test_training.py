#!/usr/bin/env python3
"""
Test the training pipeline without full training
"""

import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader

from maskrcnn_dataset import MaskRCNNDataset
from custom_maskrcnn import CustomMaskRCNN
from loss import custom_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_test_transforms():
    """Simple transforms for testing"""
    return A.Compose([
        A.Resize(height=544, width=960, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
       keypoint_params=A.KeypointParams(format='xy'))

def collate_fn(batch):
    return tuple(zip(*batch))

def test_training_step():
    """Test one training step"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Testing on {device}")
    
    # Check if data exists
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / 'data' / 'train'
    
    if not train_dir.exists():
        print(f"❌ Training data not found at {train_dir}")
        print("Please ensure you have data/train directory with images and annotations")
        return False
    
    try:
        # Dataset
        dataset = MaskRCNNDataset(train_dir, transforms=get_test_transforms())
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("❌ No samples found in dataset")
            return False
        
        # Data loader
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Model
        model = CustomMaskRCNN(num_classes=5).to(device)
        print("✅ Model created")
        
        # Test one batch
        for images, targets in loader:
            print(f"✅ Loaded batch: {len(images)} images")
            
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            print("✅ Data moved to device")
            
            # Forward pass
            model.train()
            outputs = model(images, targets)
            print("✅ Forward pass completed")
            
            # Print output shapes
            print("\n📊 Output shapes:")
            if isinstance(outputs['maskrcnn'], dict):
                print(f"  MaskRCNN losses: {list(outputs['maskrcnn'].keys())}")
            print(f"  Perspective: {outputs['perspective'].shape}")
            print(f"  Quality: {outputs['overall_quality'].shape}")
            print(f"  Text color: {outputs['text_color'].shape}")
            print(f"  Knob size: {outputs['knob_size'].shape}")
            
            # Test loss calculation
            batch_targets = {
                'perspective': torch.stack([t['perspective'] for t in targets]),
                'overall_quality': torch.stack([t['overall_quality'] for t in targets]),
                'text_color': torch.stack([t['text_color'] for t in targets]),
                'knob_size': torch.stack([t['knob_size'] for t in targets])
            }
            
            losses = custom_loss(outputs, batch_targets)
            print(f"✅ Loss calculation successful")
            
            print("\n📊 Loss values:")
            for key, value in losses.items():
                print(f"  {key}: {value.item():.4f}")
            
            # Test backward pass
            total_loss = losses['total_loss']
            total_loss.backward()
            print("✅ Backward pass successful")
            
            # Test inference mode
            model.eval()
            with torch.no_grad():
                outputs_eval = model(images, None)
                print("✅ Inference mode successful")
                
                if isinstance(outputs_eval['maskrcnn'], list):
                    print(f"  MaskRCNN predictions: {len(outputs_eval['maskrcnn'])} images")
                    for i, pred in enumerate(outputs_eval['maskrcnn']):
                        print(f"    Image {i}: {len(pred['boxes'])} detections")
            
            break  # Only test one batch
        
        print("\n🎉 All tests passed! Training pipeline is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_training_step() 