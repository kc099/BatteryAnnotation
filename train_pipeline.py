import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class BatteryDataset(Dataset):
    """Custom dataset for battery cover inspection"""
    
    def __init__(self, annotations_dir, transform=None):
        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = list(self.annotations_dir.glob("*_annotation.json"))
        self.transform = transform
        
        # Define keypoint connections for visualization
        self.keypoint_names = ['hole_' + str(i) for i in range(10)]
        
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        # Load annotation
        with open(self.annotation_files[idx], 'r') as f:
            ann = json.load(f)
        
        # Load image
        image = cv2.imread(ann['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare target
        boxes = []
        labels = []
        keypoints = []
        
        # Text region
        if ann['text_bbox']:
            bbox = ann['text_bbox']
            boxes.append([bbox['x'], bbox['y'], 
                         bbox['x'] + bbox['width'], 
                         bbox['y'] + bbox['height']])
            labels.append(1)  # text class
        
        # Plus knob
        if ann['plus_knob_center'] and ann['plus_knob_radius']:
            cx, cy = ann['plus_knob_center']
            r = ann['plus_knob_radius']
            boxes.append([cx-r, cy-r, cx+r, cy+r])
            labels.append(2)  # plus knob class
        
        # Minus knob
        if ann['minus_knob_center'] and ann['minus_knob_radius']:
            cx, cy = ann['minus_knob_center']
            r = ann['minus_knob_radius']
            boxes.append([cx-r, cy-r, cx+r, cy+r])
            labels.append(3)  # minus knob class
        
        # Holes as keypoints (pad to fixed number)
        holes = ann['holes'][:10]  # Take first 10 holes
        while len(holes) < 10:
            holes.append([0, 0])  # Pad with invisible keypoints
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Keypoints format: [x, y, visibility]
        kpts = []
        for i, (x, y) in enumerate(holes):
            if i < len(ann['holes']):
                kpts.extend([x, y, 2])  # 2 = visible
            else:
                kpts.extend([0, 0, 0])  # 0 = not visible
        
        keypoints = torch.as_tensor([kpts], dtype=torch.float32).reshape(1, -1, 3)
        
        # Additional attributes for classification head
        attributes = torch.tensor([
            float(ann.get('text_color_present', False)),
            float(ann.get('text_readable', False)),
            float(ann.get('knob_size_ratio_correct', False))
        ], dtype=torch.float32)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'attributes': attributes,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transformations
        if self.transform:
            # Albumentations transform
            transformed = self.transform(image=image, 
                                       bboxes=boxes.numpy() if len(boxes) > 0 else [],
                                       keypoints=[(kpt[0], kpt[1]) for kpt in keypoints.reshape(-1, 3).numpy()],
                                       labels=labels.numpy() if len(labels) > 0 else [])
            image = transformed['image']
            
            # Update target with transformed values
            if transformed['bboxes']:
                target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            
            if transformed['keypoints']:
                # Reconstruct keypoints tensor
                new_kpts = []
                for i, (x, y) in enumerate(transformed['keypoints']):
                    vis = keypoints.reshape(-1, 3)[i, 2]
                    new_kpts.extend([x, y, vis])
                target['keypoints'] = torch.tensor([new_kpts], dtype=torch.float32).reshape(1, -1, 3)
        
        return image, target

class MultiTaskBatteryModel(nn.Module):
    """Multi-task model for battery inspection"""
    
    def __init__(self, num_classes=4, num_keypoints=10, num_attributes=3):
        super().__init__()
        
        # Base model - Keypoint R-CNN
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        
        # RPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * 5,
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        # Initialize base model
        self.base_model = KeypointRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            num_keypoints=num_keypoints
        )
        
        # Additional attribute classification head
        self.attribute_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_attributes),
            nn.Sigmoid()
        )
        
    def forward(self, images, targets=None):
        # Get features from backbone
        features = self.backbone(images)
        
        # Get detection and keypoint predictions
        if self.training and targets is not None:
            detections = self.base_model(images, targets)
            
            # Extract features for attribute prediction
            # Use the last feature map
            feat = list(features.values())[-1]
            attributes = self.attribute_head(feat)
            
            # Add attribute loss
            if targets:
                attr_targets = torch.stack([t['attributes'] for t in targets])
                attr_loss = nn.functional.binary_cross_entropy(attributes, attr_targets)
                detections['loss_attributes'] = attr_loss
                
            return detections
        else:
            # Inference mode
            detections = self.base_model(images)
            
            # Get attribute predictions
            features = self.backbone(images)
            feat = list(features.values())[-1]
            attributes = self.attribute_head(feat)
            
            # Add attributes to predictions
            for i, det in enumerate(detections):
                det['attributes'] = attributes[i]
                
            return detections

class BatteryInspectionModule(pl.LightningModule):
    """PyTorch Lightning module for training"""
    
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Forward pass
        loss_dict = self.model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Log losses
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, prog_bar=True)
        
        self.log('train_loss', losses, prog_bar=True)
        return losses
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        # Inference
        predictions = self.model(images)
        
        # Calculate metrics (implement based on your needs)
        # Example: mAP, keypoint accuracy, attribute accuracy
        
        return predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

def get_transforms(train=True):
    """Get augmentation transforms"""
    if train:
        return A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
           keypoint_params=A.KeypointParams(format='xy'))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def train_model(data_dir, epochs=100, batch_size=4):
    """Main training function"""
    
    # Create datasets
    train_dataset = BatteryDataset(
        Path(data_dir) / 'train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = BatteryDataset(
        Path(data_dir) / 'val',
        transform=get_transforms(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    model = MultiTaskBatteryModel(
        num_classes=4,  # background, text, plus_knob, minus_knob
        num_keypoints=10,
        num_attributes=3
    )
    
    # Create Lightning module
    lightning_model = BatteryInspectionModule(model)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # Mixed precision training
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                save_top_k=3,
                mode='min'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor()
        ]
    )
    
    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    
    return lightning_model

def inference(model, image_path, confidence_threshold=0.5):
    """Run inference on a single image"""
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transforms(train=False)
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Process predictions
    results = {
        'boxes': predictions['boxes'][predictions['scores'] > confidence_threshold],
        'labels': predictions['labels'][predictions['scores'] > confidence_threshold],
        'scores': predictions['scores'][predictions['scores'] > confidence_threshold],
        'keypoints': predictions['keypoints'][0] if 'keypoints' in predictions else None,
        'attributes': predictions['attributes'].cpu().numpy() if 'attributes' in predictions else None
    }
    
    # Interpret results
    quality_checks = {
        'holes_detected': len([kp for kp in results['keypoints'] if kp[2] > 0]) if results['keypoints'] is not None else 0,
        'text_region_found': 1 in results['labels'].tolist(),
        'plus_knob_found': 2 in results['labels'].tolist(),
        'minus_knob_found': 3 in results['labels'].tolist(),
    }
    
    if results['attributes'] is not None:
        quality_checks.update({
            'white_text_present': results['attributes'][0] > 0.5,
            'text_readable': results['attributes'][1] > 0.5,
            'knob_ratio_correct': results['attributes'][2] > 0.5
        })
    
    # Determine overall quality
    is_good = (
        quality_checks['holes_detected'] >= 8 and  # At least 8 holes
        quality_checks['text_region_found'] and
        quality_checks['plus_knob_found'] and
        quality_checks['minus_knob_found'] and
        quality_checks.get('white_text_present', False) and
        quality_checks.get('knob_ratio_correct', False)
    )
    
    quality_checks['overall_quality'] = 'GOOD' if is_good else 'BAD'
    
    return results, quality_checks

# Visualization function
def visualize_predictions(image_path, results, quality_checks):
    """Visualize model predictions"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes
    class_names = ['background', 'text', 'plus_knob', 'minus_knob']
    colors = ['none', 'green', 'magenta', 'cyan']
    
    for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
        x1, y1, x2, y2 = box.tolist()
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color=colors[label], linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"{class_names[label]} ({score:.2f})", 
               color=colors[label], fontsize=10)
    
    # Draw keypoints (holes)
    if results['keypoints'] is not None:
        for i, kp in enumerate(results['keypoints'].reshape(-1, 3)):
            if kp[2] > 0:  # Visible
                circle = Circle((kp[0], kp[1]), 8, fill=False, color='red', linewidth=2)
                ax.add_patch(circle)
    
    # Add quality information
    quality_text = f"Overall: {quality_checks['overall_quality']}\n"
    for key, value in quality_checks.items():
        if key != 'overall_quality':
            quality_text += f"{key}: {value}\n"
    
    ax.text(10, 30, quality_text, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
           fontsize=10, verticalalignment='top')
    
    ax.set_title("Battery Cover Inspection Results", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    # Train model
    # model = train_model("path/to/annotated/data", epochs=100)
    
    # Or load pretrained model and run inference
    # model = torch.load("path/to/checkpoint.pth")
    # results, quality = inference(model, "path/to/test/image.jpg")
    # visualize_predictions("path/to/test/image.jpg", results, quality)
    pass