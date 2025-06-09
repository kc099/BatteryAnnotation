import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from shapely.geometry import Polygon
import json
from pathlib import Path

class ComponentQualityDataset(Dataset):
    """Dataset with separate quality labels for each component"""
    
    def __init__(self, data_dir, split='train', train_ratio=0.8, transform=None, seed=42):
        """
        Args:
            data_dir: Directory containing images and annotation files (e.g., extracted_frames_9182)
            split: 'train' or 'val'  
            train_ratio: Fraction of data for training
            transform: Data augmentation pipeline
            seed: Random seed for consistent splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Auto-discover and split data
        self.annotations = self._discover_and_split_data(train_ratio, seed)
        print(f"✅ {split.upper()} dataset: {len(self.annotations)} samples")
        
        # Component quality mapping - Fixed to handle both string formats
        self.quality_map = {
            "GOOD": 1, "good": 1, 
            "BAD": 0, "bad": 0, 
            "deformed": 0, "blocked": 0,  # Hole-specific bad qualities
            "UNKNOWN": -1, "unknown": -1
        }
        
    def _discover_and_split_data(self, train_ratio, seed):
        """Auto-discover annotation files and split train/val"""
        print(f"🔍 Discovering data in {self.data_dir}")
        
        # Find all annotation files
        annotation_files = list(self.data_dir.glob("*_enhanced_annotation.json"))
        
        if not annotation_files:
            raise ValueError(f"No annotation files found in {self.data_dir}")
        
        print(f"   Found {len(annotation_files)} annotation files")
        
        # Load and validate annotations
        valid_annotations = []
        for ann_file in annotation_files:
            # Find corresponding image
            image_file = ann_file.with_name(ann_file.name.replace('_enhanced_annotation.json', '.jpg'))
            
            if not image_file.exists():
                print(f"   ⚠️  Missing image for {ann_file.name}")
                continue
                
            # Load annotation
            try:
                with open(ann_file, 'r') as f:
                    ann = json.load(f)
                
                # Add file paths
                ann['annotation_file'] = str(ann_file)
                ann['image_file'] = str(image_file)
                
                # Filter by confidence
                if ann.get('confidence_score', 1.0) > 0.7:
                    valid_annotations.append(ann)
                    
            except Exception as e:
                print(f"   ⚠️  Error loading {ann_file.name}: {e}")
                continue
        
        print(f"   ✅ {len(valid_annotations)} valid samples (confidence > 0.7)")
        
        # Deterministic split
        import random
        random.seed(seed)
        indices = list(range(len(valid_annotations)))
        random.shuffle(indices)
        
        split_idx = int(len(valid_annotations) * train_ratio)
        
        if self.split == 'train':
            selected_indices = indices[:split_idx]
        else:  # val
            selected_indices = indices[split_idx:]
        
        return [valid_annotations[i] for i in selected_indices]

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image - Use stored image path
        img_path = ann['image_file']
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create segmentation masks (6 channels)
        masks = self.create_masks(ann, h, w)
        
        # Extract geometric features
        features = self.extract_features(ann)
        
        # Component quality labels - Fixed to handle different annotation formats
        hole_quality = self._get_quality_label(ann, 'hole_quality', 'hole_qualities')
        text_quality = self._get_quality_label(ann, 'text_quality')
        knob_quality = self._get_quality_label(ann, 'knob_quality') 
        surface_quality = self._get_quality_label(ann, 'surface_quality')
        overall_quality = self._get_quality_label(ann, 'overall_quality')
        
        # Sample weight based on annotation confidence and quality
        weight = ann.get('confidence_score', 1.0)
        if overall_quality == 0:  # Bad samples get higher weight
            weight *= 2.0
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=masks)
            image = transformed['image']
            masks = transformed['mask']
        
        return {
            'image': image,
            'masks': masks.permute(2, 0, 1),  # HWC to CHW
            'features': features,
            'hole_quality': torch.tensor(hole_quality, dtype=torch.float32),
            'text_quality': torch.tensor(text_quality, dtype=torch.float32),
            'knob_quality': torch.tensor(knob_quality, dtype=torch.float32),
            'surface_quality': torch.tensor(surface_quality, dtype=torch.float32),
            'overall_quality': torch.tensor(overall_quality, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'has_perspective': torch.tensor(ann.get('perspective_points') is not None, dtype=torch.float32)
        }
    

    
    def _get_quality_label(self, ann, quality_key, qualities_array_key=None):
        """Extract and normalize quality labels from annotations"""
        # First try the direct quality key
        if quality_key in ann:
            quality_val = ann[quality_key]
            return self.quality_map.get(quality_val, -1)
        
        # If there's an array key (like hole_qualities), use majority vote
        if qualities_array_key and qualities_array_key in ann:
            qualities = ann[qualities_array_key]
            if qualities:
                # Count quality types
                quality_counts = {}
                for q in qualities:
                    mapped_val = self.quality_map.get(q, -1)
                    quality_counts[mapped_val] = quality_counts.get(mapped_val, 0) + 1
                
                # Return majority quality (prioritize bad over good)
                if 0 in quality_counts:  # Any bad quality
                    return 0
                elif 1 in quality_counts:  # All good
                    return 1
                else:
                    return -1  # Unknown
        
        return -1  # Default to unknown
    
    def create_masks(self, ann, h, w):
        """Create segmentation masks for different components
        
        Creates 6-channel mask from 4 polygon types in JSON:
        - hole_polygons → 3 channels (separated by quality: good/deformed/blocked)
        - text_polygon → 1 channel
        - plus_knob_polygon → 1 channel  
        - minus_knob_polygon → 1 channel
        Total: 6 channels
        """
        masks = np.zeros((h, w, 6), dtype=np.float32)
        
        # HOLE MASKS (3 channels based on quality)
        # Channel 0: Good holes
        # Channel 1: Deformed holes  
        # Channel 2: Blocked holes
        hole_polygons = ann.get('hole_polygons', [])
        hole_qualities = ann.get('hole_qualities', [])
        
        for i, polygon in enumerate(hole_polygons):
            if len(polygon) >= 3:
                points = np.array(polygon, dtype=np.int32)
                # Get quality for this hole
                quality = hole_qualities[i] if i < len(hole_qualities) else 'good'
                channel = {"good": 0, "deformed": 1, "blocked": 2}.get(quality, 0)
                
                # Fix: Create contiguous array for OpenCV
                channel_mask = np.ascontiguousarray(masks[:, :, channel])
                cv2.fillPoly(channel_mask, [points], 1)
                masks[:, :, channel] = channel_mask
        
        # KNOB AND TEXT MASKS (1 channel each)
        # Channel 3: Text region
        if ann.get('text_polygon'):
            points = np.array(ann['text_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 3])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 3] = channel_mask
        
        # Channel 4: Plus knob
        if ann.get('plus_knob_polygon'):
            points = np.array(ann['plus_knob_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 4])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 4] = channel_mask
        
        # Channel 5: Minus knob
        if ann.get('minus_knob_polygon'):
            points = np.array(ann['minus_knob_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 5])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 5] = channel_mask
        
        return masks
    
    def extract_features(self, ann):
        """Extract hand-crafted features - Fixed for annotation compatibility"""
        features = []
        
        # Hole features
        hole_polygons = ann.get('hole_polygons', [])
        num_holes = len(hole_polygons)
        features.append(num_holes / 12.0)  # Normalized by expected count
        
        # Hole quality distribution
        qualities = ann.get('hole_qualities', [])
        total_holes = max(len(qualities), 1)
        features.extend([
            qualities.count('good') / total_holes,
            qualities.count('deformed') / total_holes,
            qualities.count('blocked') / total_holes
        ])
        
        # Average hole circularity
        circularities = []
        for polygon in hole_polygons:
            if len(polygon) >= 3:
                try:
                    poly = Polygon(polygon)
                    if poly.length > 0 and poly.area > 0:
                        circularity = 4 * np.pi * poly.area / (poly.length ** 2)
                        circularities.append(circularity)
                except Exception:
                    # Skip invalid polygons
                    continue
        
        avg_circularity = np.mean(circularities) if circularities else 0
        features.append(avg_circularity)
        
        # Text features
        has_text = float(ann.get('text_polygon') is not None)
        features.extend([
            has_text,
            float(ann.get('text_color_present', False)),
            float(ann.get('text_readable', False))
        ])
        
        # Knob features
        has_plus = float(ann.get('plus_knob_polygon') is not None)
        has_minus = float(ann.get('minus_knob_polygon') is not None)
        
        # Knob size comparison - Fixed to handle missing area values
        knob_size_correct = 0.0
        if ann.get('plus_knob_area') and ann.get('minus_knob_area'):
            knob_size_correct = float(ann['plus_knob_area'] > ann['minus_knob_area'])
        elif 'knob_size_correct' in ann:
            knob_size_correct = float(ann['knob_size_correct'])
        else:
            # Calculate areas if polygons exist but areas are missing
            if ann.get('plus_knob_polygon') and ann.get('minus_knob_polygon'):
                try:
                    plus_poly = Polygon(ann['plus_knob_polygon'])
                    minus_poly = Polygon(ann['minus_knob_polygon'])
                    knob_size_correct = float(plus_poly.area > minus_poly.area)
                except Exception:
                    knob_size_correct = 0.0
        
        features.extend([has_plus, has_minus, knob_size_correct])
        
        # Perspective correction indicator
        has_perspective = float(ann.get('perspective_points') is not None)
        features.append(has_perspective)
        
        return torch.tensor(features, dtype=torch.float32)

def get_training_augmentations():
    """Get augmentations for training"""
    return A.Compose([
        # Resize preserving aspect ratio (1920x1080 -> 960x544, exactly 1/2 scale)
        A.Resize(544, 960),
        
        # Geometric transforms (mild to preserve polygon shapes)
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Perspective (simulate different viewing angles)
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.8),
        
        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(5, 25)),
            A.ISONoise(color_shift=(0.01, 0.05)),
        ], p=0.2),
        
        # Blur (simulate focus issues)
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.1),
        
        # Normalize with battery-specific values (computed from dataset)
        A.Normalize(mean=[0.4045, 0.4045, 0.4045], std=[0.2256, 0.2254, 0.2782]),
        ToTensorV2()
    ])

def get_validation_augmentations():
    """Get augmentations for validation"""
    return A.Compose([
        # Resize preserving aspect ratio (1920x1080 -> 960x544, exactly 1/2 scale)
        A.Resize(544, 960),
        A.Normalize(mean=[0.4045, 0.4045, 0.4045], std=[0.2256, 0.2254, 0.2782]),
        ToTensorV2()
    ]) 