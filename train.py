#!/usr/bin/env python3
"""
Battery Quality Training Pipeline with CustomMaskRCNN

Usage:
    python train.py --epochs 50 --batch_size 4
"""

import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from maskrcnn_dataset import MaskRCNNDataset
from custom_maskrcnn import CustomMaskRCNN
from loss import custom_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Evaluation metrics
from torchvision.ops import box_iou
from sklearn.metrics import f1_score, accuracy_score

def get_transforms(train=True):
    """Get augmentation pipeline for training/validation"""
    if train:
        return A.Compose([
            A.Resize(height=544, width=960, p=1.0),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
           keypoint_params=A.KeypointParams(format='xy'))
    else:
        return A.Compose([
            A.Resize(height=544, width=960, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
           keypoint_params=A.KeypointParams(format='xy'))

def collate_fn(batch):
    """Custom collate function for variable-sized inputs"""
    return tuple(zip(*batch))

def calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """Calculate mean Average Precision"""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    # Calculate IoU between all pred and gt boxes
    ious = box_iou(pred_boxes, gt_boxes)
    
    # For each prediction, find best matching GT
    max_ious, matched_gt = torch.max(ious, dim=1)
    
    # Count true positives (IoU > threshold and correct class)
    tp = 0
    fp = 0
    
    for i, (iou, gt_idx) in enumerate(zip(max_ious, matched_gt)):
        if iou > iou_threshold and pred_labels[i] == gt_labels[gt_idx]:
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def evaluate_model(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    total_losses = {}
    all_precisions = []
    all_recalls = []
    
    # For classification metrics
    all_quality_preds = []
    all_quality_targets = []
    all_text_preds = []
    all_text_targets = []
    all_knob_preds = []
    all_knob_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images, targets)
            
            # Calculate losses
            batch_targets = {
                'perspective': torch.stack([t['perspective'] for t in targets]),
                'overall_quality': torch.stack([t['overall_quality'] for t in targets]),
                'text_color': torch.stack([t['text_color'] for t in targets]),
                'knob_size': torch.stack([t['knob_size'] for t in targets])
            }
            
            losses = custom_loss(outputs, batch_targets)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value.item()
            
            # Collect predictions for metrics
            # Overall quality is now rule-based (0=good, 1=bad)
            quality_preds = outputs['overall_quality'].cpu().numpy()
            if quality_preds.ndim == 0:
                quality_preds = [quality_preds.item()]
            else:
                quality_preds = quality_preds.tolist()
            all_quality_preds.extend(quality_preds)
            
            quality_targets = batch_targets['overall_quality'].cpu().numpy()
            if quality_targets.ndim == 0:
                quality_targets = [quality_targets.item()]
            else:
                quality_targets = quality_targets.tolist()
            all_quality_targets.extend(quality_targets)
            
            # Text color and knob size are now probabilities (0-1)
            text_preds = (outputs['text_color'] > 0.5).float().squeeze()
            if text_preds.ndim == 0:
                text_preds = [text_preds.item()]
            else:
                text_preds = text_preds.tolist()
            all_text_preds.extend(text_preds)
            
            text_targets = batch_targets['text_color'].cpu().numpy()
            if text_targets.ndim == 0:
                text_targets = [text_targets.item()]
            else:
                text_targets = text_targets.tolist()
            all_text_targets.extend(text_targets)
            
            knob_preds = (outputs['knob_size'] > 0.5).float().squeeze()
            if knob_preds.ndim == 0:
                knob_preds = [knob_preds.item()]
            else:
                knob_preds = knob_preds.tolist()
            all_knob_preds.extend(knob_preds)
            
            knob_targets = batch_targets['knob_size'].cpu().numpy()
            if knob_targets.ndim == 0:
                knob_targets = [knob_targets.item()]
            else:
                knob_targets = knob_targets.tolist()
            all_knob_targets.extend(knob_targets)
            
            # Detection metrics (simplified)
            maskrcnn_outputs = outputs['maskrcnn']
            for i, (output, target) in enumerate(zip(maskrcnn_outputs, targets)):
                if len(output['boxes']) > 0 and len(target['boxes']) > 0:
                    precision, recall = calculate_map(
                        output['boxes'], output['scores'], output['labels'],
                        target['boxes'], target['labels']
                    )
                    all_precisions.append(precision)
                    all_recalls.append(recall)
    
    # Average losses
    avg_losses = {k: v / len(data_loader) for k, v in total_losses.items()}
    
    # Calculate metrics
    quality_f1 = f1_score(all_quality_targets, all_quality_preds, average='weighted')
    quality_acc = accuracy_score(all_quality_targets, all_quality_preds)
    text_f1 = f1_score(all_text_targets, all_text_preds, average='binary')
    knob_f1 = f1_score(all_knob_targets, all_knob_preds, average='binary')
    
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    
    return {
        'losses': avg_losses,
        'quality_f1': quality_f1,
        'quality_accuracy': quality_acc,
        'text_f1': text_f1,
        'knob_f1': knob_f1,
        'detection_precision': avg_precision,
        'detection_recall': avg_recall
    }

def train_model(train_dir, val_dir, epochs=50, batch_size=4, lr=1e-4, num_workers=4):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on device: {device}")
    
    # Datasets
    train_dataset = MaskRCNNDataset(train_dir, transforms=get_transforms(train=True))
    val_dataset = MaskRCNNDataset(val_dir, transforms=get_transforms(train=False))
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Model
    model = CustomMaskRCNN(num_classes=5).to(device)  # 4 classes + background
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    print(f"\n🎯 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_losses = {}
        
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images, targets)
            
            # Prepare targets for custom loss
            batch_targets = {
                'perspective': torch.stack([t['perspective'] for t in targets]),
                'overall_quality': torch.stack([t['overall_quality'] for t in targets]),
                'text_color': torch.stack([t['text_color'] for t in targets]),
                'knob_size': torch.stack([t['knob_size'] for t in targets])
            }
            
            # Calculate loss
            losses = custom_loss(outputs, batch_targets)
            total_loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in train_losses:
                    train_losses[key] = 0
                train_losses[key] += value.item()
        
        # Average training losses
        avg_train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {avg_train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['losses']['total_loss']:.4f}")
        print(f"Quality F1: {val_metrics['quality_f1']:.3f} | Text F1: {val_metrics['text_f1']:.3f} | Knob F1: {val_metrics['knob_f1']:.3f}")
        print(f"Detection P: {val_metrics['detection_precision']:.3f} | R: {val_metrics['detection_recall']:.3f}")
        
        # Save best model
        current_f1 = (val_metrics['quality_f1'] + val_metrics['text_f1'] + val_metrics['knob_f1']) / 3
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, 'best_custom_maskrcnn.pth')
            print(f"💾 New best model saved! F1: {best_f1:.3f}")
    
    print(f"\n✅ Training completed! Best F1: {best_f1:.3f}")
    return 'best_custom_maskrcnn.pth'

def main():
    parser = argparse.ArgumentParser(description='Train Custom Mask R-CNN for Battery Quality')
    parser.add_argument('--data_dir', type=str, default='data', help='Base data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    train_dir = base_dir / args.data_dir / 'train'
    val_dir = base_dir / args.data_dir / 'valid'
    
    # Train
    best_model_path = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers
    )
    
    print(f"🎉 Best model saved at: {best_model_path}")

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main() 

# # 1. Prepare dataset (when adding new data)
# python BatteryAnnotation/prepare_dataset.py

# # 2. Train model
# python BatteryAnnotation/train.py --epochs 50 --batch_size 4

# # 3. Run inference
# python BatteryAnnotation/inference.py --model best_custom_maskrcnn.pth --data_dir extracted_frames_9213 --save_results

# # 4. Test components (optional)
# python BatteryAnnotation/test_custom_maskrcnn.py
# python BatteryAnnotation/test_training.py