#!/usr/bin/env python3

import torch
import numpy as np
from custom_maskrcnn import CustomMaskRCNN

def test_model_structure():
    """Test model with synthetic data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧪 Testing CustomMaskRCNN on {device}")
    
    # Create model
    model = CustomMaskRCNN(num_classes=5).to(device)
    print("✅ Model created")
    
    # Create synthetic inputs
    batch_size = 2
    height, width = 544, 960
    
    # Synthetic images
    images = [torch.randn(3, height, width).to(device) for _ in range(batch_size)]
    
    # Synthetic targets for training
    targets = []
    for i in range(batch_size):
        # Create some dummy objects
        num_objects = 3
        boxes = torch.tensor([
            [50, 50, 150, 150],
            [200, 200, 300, 350],
            [400, 100, 500, 200]
        ], dtype=torch.float32).to(device)
        
        labels = torch.tensor([1, 2, 3], dtype=torch.int64).to(device)
        masks = torch.zeros(num_objects, height, width, dtype=torch.uint8).to(device)
        
        # Fill in some mask regions
        masks[0, 50:150, 50:150] = 1
        masks[1, 200:350, 200:300] = 1
        masks[2, 100:200, 400:500] = 1
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([i]).to(device),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(num_objects, dtype=torch.int64).to(device),
            'perspective': torch.tensor([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8], dtype=torch.float32).to(device),
            'overall_quality': torch.tensor(0, dtype=torch.long).to(device),  # GOOD
            'text_color': torch.tensor(1.0, dtype=torch.float32).to(device),
            'knob_size': torch.tensor(1.0, dtype=torch.float32).to(device)
        }
        targets.append(target)
    
    print("✅ Synthetic data created")
    
    # Test training mode
    model.train()
    outputs = model(images, targets)
    
    print("\n📊 Training mode outputs:")
    print(f"  MaskRCNN: {type(outputs['maskrcnn'])}")
    if isinstance(outputs['maskrcnn'], dict):
        print(f"    Losses: {list(outputs['maskrcnn'].keys())}")
    print(f"  Perspective: {outputs['perspective'].shape}")
    print(f"  Quality: {outputs['overall_quality'].shape}")
    print(f"  Text color: {outputs['text_color'].shape}")
    print(f"  Knob size: {outputs['knob_size'].shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        outputs = model(images, None)
    
    print("\n📊 Inference mode outputs:")
    print(f"  MaskRCNN: {type(outputs['maskrcnn'])}")
    if isinstance(outputs['maskrcnn'], list):
        print(f"    Predictions for {len(outputs['maskrcnn'])} images")
        for i, pred in enumerate(outputs['maskrcnn']):
            print(f"      Image {i}: {len(pred['boxes'])} detections")
    print(f"  Perspective: {outputs['perspective'].shape}")
    print(f"  Quality: {outputs['overall_quality'].shape}")
    print(f"  Text color: {outputs['text_color'].shape}")
    print(f"  Knob size: {outputs['knob_size'].shape}")
    
    print("\n🎉 CustomMaskRCNN test completed successfully!")
    
    # Check data directory
    from pathlib import Path
    data_dir = Path('../data/train')
    if data_dir.exists():
        from maskrcnn_dataset import MaskRCNNDataset
        dataset = MaskRCNNDataset(data_dir)
        print(f"\n📂 Real dataset available: {len(dataset)} samples")
        
        if len(dataset) > 0:
            print("Testing with real data...")
            try:
                image, target = dataset[0]
                print(f"  Real image shape: {image.shape}")
                print(f"  Real target keys: {list(target.keys())}")
                print("✅ Real data test passed")
            except Exception as e:
                print(f"❌ Real data test failed: {e}")
    else:
        print(f"\n📂 No real dataset found at {data_dir}")

if __name__ == "__main__":
    test_model_structure() 