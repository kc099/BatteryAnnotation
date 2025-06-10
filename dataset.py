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
    """
    Dataset that loads images and annotations from a single, pre-split directory.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing the pre-split images and annotation files 
                      (e.g., 'data/train' or 'data/valid')
            transform: Data augmentation pipeline
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Discover all annotation files in the directory
        self.annotations = self._discover_data()
        print(f"✅ Loaded {len(self.annotations)} samples from {self.data_dir.name}")
        
        self.quality_map = {
            "GOOD": 1, "good": 1, 
            "BAD": 0, "bad": 0, 
            "deformed": 0, "blocked": 0,
            "UNKNOWN": -1, "unknown": -1
        }
        
    def _discover_data(self):
        """Finds all valid annotation files and their corresponding images."""
        valid_annotations = []
        for ann_file in self.data_dir.glob("*_enhanced_annotation.json"):
            image_file = ann_file.with_name(ann_file.name.replace('_enhanced_annotation.json', '.jpg'))
            
            if not image_file.exists():
                continue
                
            try:
                with open(ann_file, 'r') as f:
                    ann = json.load(f)
                
                # Add file paths
                ann['annotation_file'] = str(ann_file)
                ann['image_file'] = str(image_file)
                valid_annotations.append(ann)
                    
            except Exception:
                continue
        
        if not valid_annotations:
            raise ValueError(f"No valid annotation files found in {self.data_dir}!")
            
        return valid_annotations

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
        
        # Create segmentation masks (now only 4 channels - removing deformed/blocked holes)
        masks = self.create_masks(ann, h, w)
        
        # Extract geometric features
        features = self.extract_features(ann)
        
        # Get bounding boxes for ROI-Align
        hole_boxes = self._polygons_to_boxes(ann.get('hole_polygons', []))
        text_boxes = self._polygons_to_boxes([ann.get('text_polygon', [])])
        
        # Combine knob polygons for a single "knob" class
        knob_polygons = []
        if 'plus_knob_polygon' in ann and ann['plus_knob_polygon']:
             knob_polygons.append(ann['plus_knob_polygon'])
        if 'minus_knob_polygon' in ann and ann['minus_knob_polygon']:
             knob_polygons.append(ann['minus_knob_polygon'])
        knob_boxes = self._polygons_to_boxes(knob_polygons)
        
        # Component quality labels - Fixed to handle different annotation formats
        hole_quality_val = self._get_quality_label(ann, 'hole_quality', 'hole_qualities')
        text_quality_val = self._get_quality_label(ann, 'text_quality')
        knob_quality_val = self._get_quality_label(ann, 'knob_quality')
        surface_quality_val = self._get_quality_label(ann, 'surface_quality')
        overall_quality_val = self._get_quality_label(ann, 'overall_quality')

        # Create per-instance quality targets for the new model
        # Use per-hole qualities if available, otherwise fallback to global
        if 'hole_qualities' in ann and ann['hole_qualities']:
            # Map each quality to numeric value using quality_map
            hole_quality = torch.tensor([
                self.quality_map.get(q, -1) for q in ann['hole_qualities']
            ], dtype=torch.float32)
        else:
            hole_quality = torch.full((len(hole_boxes),), hole_quality_val, dtype=torch.float32)
        text_quality = torch.full((len(text_boxes),), text_quality_val, dtype=torch.float32)
        knob_quality = torch.full((len(knob_boxes),), knob_quality_val, dtype=torch.float32)
        
        # Quality confidence weights - Fixed calculation for missing confidence values
        weight = ann.get('confidence_score', 1.0)
        if isinstance(weight, (list, tuple)):
            weight = np.mean(weight)
        
        # Extract perspective points if available
        perspective_points = ann.get('perspective_points', [])
        if perspective_points and len(perspective_points) == 4:
            # Flatten 4 points to 8 coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
            perspective_target = torch.tensor([coord for point in perspective_points for coord in point], dtype=torch.float32)
            # Normalize to [0,1] range
            perspective_target[0::2] /= w  # x coordinates
            perspective_target[1::2] /= h  # y coordinates
        else:
            # No perspective points available
            perspective_target = torch.zeros(8, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            # Apply transform to both image and masks
            transformed = self.transform(image=image, mask=masks)
            image = transformed['image']
            masks = transformed['mask']
            
            # If masks is now a tensor, convert to proper format
            if isinstance(masks, torch.Tensor):
                masks = masks.permute(2, 0, 1)  # HWC -> CHW for consistency
            else:
                masks = torch.from_numpy(masks).permute(2, 0, 1)  # HWC -> CHW
                
            # Update perspective points for new image size
            if torch.any(perspective_target > 0):
                new_h, new_w = image.shape[-2:] if isinstance(image, torch.Tensor) else image.shape[:2]
                perspective_target[0::2] *= new_w  # Scale x coordinates
                perspective_target[1::2] *= new_h  # Scale y coordinates
        else:
            # Convert to tensor without transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            masks = torch.from_numpy(masks).permute(2, 0, 1)
        
        return {
            'image': image,
            'masks': masks,
            'features': features,
            'hole_quality': hole_quality,
            'text_quality': text_quality,
            'knob_quality': knob_quality,
            'surface_quality': torch.tensor(surface_quality_val, dtype=torch.float32),
            'overall_quality': torch.tensor(overall_quality_val, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'perspective_target': perspective_target,
            # Data for new ComponentROIModel
            'hole_boxes': hole_boxes,
            'text_boxes': text_boxes,
            'knob_boxes': knob_boxes,
            'image_path': img_path
        }
    
    def _polygons_to_boxes(self, polygons):
        """Convert a list of polygons to a tensor of bounding boxes."""
        boxes = []
        if not polygons:
            return torch.empty((0, 4), dtype=torch.float32)
            
        for polygon in polygons:
            if not polygon or len(polygon) < 3:
                continue
            
            poly_np = np.array(polygon)
            min_x, min_y = np.min(poly_np, axis=0)
            max_x, max_y = np.max(poly_np, axis=0)
            boxes.append([min_x, min_y, max_x, max_y])
            
        if not boxes:
            return torch.empty((0, 4), dtype=torch.float32)
            
        return torch.tensor(boxes, dtype=torch.float32)

    def _validate_annotation(self, ann, filename):
        """Validate that annotation has required fields and is not empty"""
        
        # Check if annotation is completely empty
        if not ann or len(ann) == 0:
            print(f"   ⚠️  Empty annotation: {filename}")
            return False
        
        # Essential fields that should exist
        required_checks = []
        warnings = []
        
        # Check hole annotations
        hole_polygons = ann.get('hole_polygons', [])
        hole_qualities = ann.get('hole_qualities', [])
        
        if not hole_polygons:
            warnings.append("No hole polygons")
        elif len(hole_polygons) != len(hole_qualities):
            warnings.append(f"Hole count mismatch: {len(hole_polygons)} polygons vs {len(hole_qualities)} qualities")
        else:
            # Check if hole polygons are valid
            valid_holes = 0
            for i, polygon in enumerate(hole_polygons):
                if isinstance(polygon, list) and len(polygon) >= 3:
                    valid_holes += 1
            if valid_holes == 0:
                warnings.append("No valid hole polygons")
        
        # Check text annotation
        if not ann.get('text_polygon'):
            warnings.append("Missing text_polygon")
        
        # Check knob annotations
        if not ann.get('plus_knob_polygon'):
            warnings.append("Missing plus_knob_polygon")
        if not ann.get('minus_knob_polygon'):
            warnings.append("Missing minus_knob_polygon")
        
        # Check quality labels
        quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'surface_quality', 'overall_quality']
        missing_qualities = [field for field in quality_fields if field not in ann]
        if missing_qualities:
            warnings.append(f"Missing quality labels: {missing_qualities}")
        
        # Count how many critical components are missing
        critical_missing = 0
        if not hole_polygons:
            critical_missing += 1
        if not ann.get('text_polygon'):
            critical_missing += 1
        if not ann.get('plus_knob_polygon') or not ann.get('minus_knob_polygon'):
            critical_missing += 1
        
        # Reject if too many critical components missing
        if critical_missing >= 2:
            print(f"   ❌ Rejected {filename}: Too many missing components ({', '.join(warnings)})")
            return False
        
        # Accept but warn about minor issues
        if warnings:
            print(f"   ⚠️  {filename}: {', '.join(warnings[:2])}{'...' if len(warnings) > 2 else ''}")
        
        return True

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
        
        Creates 4-channel mask from 4 polygon types in JSON:
        - hole_polygons → 1 channel (only good holes, no deformed/blocked in dataset)
        - text_polygon → 1 channel
        - plus_knob_polygon → 1 channel  
        - minus_knob_polygon → 1 channel
        Total: 4 channels
        """
        masks = np.zeros((h, w, 4), dtype=np.float32)
        
        # HOLE MASKS (1 channel - only good holes)
        # Channel 0: All holes (since dataset only has good holes)
        hole_polygons = ann.get('hole_polygons', [])
        
        for polygon in hole_polygons:
            if len(polygon) >= 3:
                points = np.array(polygon, dtype=np.int32)
                # All holes go to channel 0 (good holes)
                channel_mask = np.ascontiguousarray(masks[:, :, 0])
                cv2.fillPoly(channel_mask, [points], 1)
                masks[:, :, 0] = channel_mask
        
        # KNOB AND TEXT MASKS (1 channel each)
        # Channel 1: Text region
        if ann.get('text_polygon'):
            points = np.array(ann['text_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 1])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 1] = channel_mask
        
        # Channel 2: Plus knob
        if ann.get('plus_knob_polygon'):
            points = np.array(ann['plus_knob_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 2])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 2] = channel_mask
        
        # Channel 3: Minus knob
        if ann.get('minus_knob_polygon'):
            points = np.array(ann['minus_knob_polygon'], dtype=np.int32)
            channel_mask = np.ascontiguousarray(masks[:, :, 3])
            cv2.fillPoly(channel_mask, [points], 1)
            masks[:, :, 3] = channel_mask
        
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