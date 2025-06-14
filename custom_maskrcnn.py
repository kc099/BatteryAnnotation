import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import roi_align
import numpy as np

class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # num_classes: 4 masks + background (plus_knob=1, minus_knob=2, text_area=3, hole=4)
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Perspective head - uses global features (still valid for perspective correction)
        self.perspective_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8)
        )
        
        # Text color head - uses text region features
        self.text_color_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def extract_region_features(self, feature_map, boxes, output_size=7):
        """Extract features from specific regions using RoI Align"""
        if len(boxes) == 0:
            return torch.zeros((0, feature_map.shape[1], output_size, output_size), 
                             device=feature_map.device, dtype=feature_map.dtype)
        
        # Add batch indices (all from batch 0 for single image)
        batch_indices = torch.zeros(len(boxes), 1, device=boxes.device, dtype=boxes.dtype)
        rois = torch.cat([batch_indices, boxes], dim=1)
        
        # RoI Align to extract region features
        spatial_scale = 1.0 / 32.0  # C5 features are 1/32 of input size
        roi_features = roi_align(feature_map, rois, output_size, spatial_scale)
        return roi_features

    def analyze_text_color_from_mask(self, image, text_mask, threshold=0.7):
        """
        Analyze if text is white by examining pixel values in the text mask region
        Returns probability that text is white
        """
        if text_mask.sum() == 0:
            return torch.tensor(0.0, device=image.device)
        
        # Get pixels inside the text mask
        mask_bool = text_mask > 0.5
        if not mask_bool.any():
            return torch.tensor(0.0, device=image.device)
        
        # Extract RGB values from masked region
        text_pixels = image[:, mask_bool]  # [3, N] where N is number of text pixels
        
        # Calculate brightness (average of RGB channels)
        brightness = text_pixels.mean(dim=0)  # [N]
        
        # White text should have high brightness values (close to 1.0)
        white_ratio = (brightness > threshold).float().mean()
        
        return white_ratio

    def spatial_knob_size_comparison(self, plus_boxes, minus_boxes, plus_masks, minus_masks, hole_boxes, device):
        """
        Spatially-aware knob size comparison:
        1. Minus knob should be closer to hole
        2. Plus knob should be larger than minus knob
        3. If no hole detected, sample is automatically bad
        """
        if len(hole_boxes) == 0:
            # No hole detected = bad sample
            return torch.tensor(0.0, device=device)  # Bad
        
        if len(plus_boxes) == 0 or len(minus_boxes) == 0:
            # Missing knobs = bad sample
            return torch.tensor(0.0, device=device)  # Bad
        
        # Get hole center
        hole_center = (hole_boxes[0][:2] + hole_boxes[0][2:]) / 2  # [x, y]
        
        # Get knob centers and areas
        plus_center = (plus_boxes[0][:2] + plus_boxes[0][2:]) / 2
        minus_center = (minus_boxes[0][:2] + minus_boxes[0][2:]) / 2
        
        # Calculate distances to hole
        plus_to_hole_dist = torch.norm(plus_center - hole_center)
        minus_to_hole_dist = torch.norm(minus_center - hole_center)
        
        # Minus knob should be closer to hole (spatial constraint)
        spatial_constraint_met = minus_to_hole_dist < plus_to_hole_dist
        
        # Calculate actual areas from masks
        plus_area = plus_masks[0].sum().float()
        minus_area = minus_masks[0].sum().float()
        
        # Plus should be larger than minus
        size_constraint_met = plus_area > minus_area
        
        # Both constraints must be met for good knob configuration
        knob_good = spatial_constraint_met and size_constraint_met
        
        return torch.tensor(1.0 if knob_good else 0.0, device=device)

    def compute_overall_quality(self, knob_size_good, text_color_good, hole_detected, device):
        """
        Rule-based overall quality assessment:
        - Good: hole detected AND knob sizes correct AND text is white
        - Bad: any of the above conditions fail
        """
        hole_good = hole_detected > 0  # At least one hole detected
        
        overall_good = hole_good and (knob_size_good > 0.5) and (text_color_good > 0.5)
        
        if overall_good:
            return torch.tensor(0, dtype=torch.long, device=device)  # GOOD
        else:
            return torch.tensor(1, dtype=torch.long, device=device)  # BAD

    def forward(self, images, targets=None):
        # Standard Mask R-CNN outputs
        maskrcnn_out = self.model(images, targets) if self.training else self.model(images)
        
        # Get backbone features
        if isinstance(images, list):
            imgs = torch.stack(images)
        else:
            imgs = images
        
        backbone_out = self.model.backbone.body(imgs)
        c5_features = backbone_out['3']  # [B, 2048, H/32, W/32]
        
        batch_size = c5_features.shape[0]
        
        # Initialize outputs
        perspective_outputs = []
        text_color_outputs = []
        knob_size_outputs = []
        overall_quality_outputs = []
        
        for batch_idx in range(batch_size):
            # Global features for perspective (GAP still appropriate here)
            global_features = torch.mean(c5_features[batch_idx], dim=[1, 2])  # [2048]
            perspective = self.perspective_head(global_features)
            perspective_outputs.append(perspective)
            
            if self.training:
                # During training, use ground truth for region extraction
                target = targets[batch_idx]
                boxes = target['boxes']
                labels = target['labels']
                masks = target['masks']
                
                # Separate by class
                plus_mask = labels == 1
                minus_mask = labels == 2
                text_mask = labels == 3
                hole_mask = labels == 4
                
                plus_boxes = boxes[plus_mask] if plus_mask.any() else torch.empty((0, 4), device=boxes.device)
                minus_boxes = boxes[minus_mask] if minus_mask.any() else torch.empty((0, 4), device=boxes.device)
                text_boxes = boxes[text_mask] if text_mask.any() else torch.empty((0, 4), device=boxes.device)
                hole_boxes = boxes[hole_mask] if hole_mask.any() else torch.empty((0, 4), device=boxes.device)
                
                plus_masks = masks[plus_mask] if plus_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
                minus_masks = masks[minus_mask] if minus_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
                text_masks = masks[text_mask] if text_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
                
            else:
                # During inference, use predictions
                pred = maskrcnn_out[batch_idx]
                boxes = pred['boxes']
                labels = pred['labels']
                masks = pred['masks']
                scores = pred['scores']
                
                # Filter by confidence
                keep = scores > 0.5
                boxes = boxes[keep]
                labels = labels[keep]
                masks = masks[keep]
                
                # Separate by class
                plus_mask = labels == 1
                minus_mask = labels == 2
                text_mask = labels == 3
                hole_mask = labels == 4
                
                plus_boxes = boxes[plus_mask] if plus_mask.any() else torch.empty((0, 4), device=boxes.device)
                minus_boxes = boxes[minus_mask] if minus_mask.any() else torch.empty((0, 4), device=boxes.device)
                text_boxes = boxes[text_mask] if text_mask.any() else torch.empty((0, 4), device=boxes.device)
                hole_boxes = boxes[hole_mask] if hole_mask.any() else torch.empty((0, 4), device=boxes.device)
                
                plus_masks = masks[plus_mask] if plus_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
                minus_masks = masks[minus_mask] if minus_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
                text_masks = masks[text_mask] if text_mask.any() else torch.empty((0, *masks.shape[1:]), device=masks.device)
            
            # Text color analysis using direct mask analysis
            if len(text_masks) > 0:
                # Use the original image for color analysis
                orig_image = imgs[batch_idx]  # [3, H, W]
                # Resize mask to match image size
                # Ensure proper dimensions for interpolation: [N, C, H, W]
                mask_for_interp = text_masks[0].float()  # [H, W]
                if mask_for_interp.dim() == 2:
                    mask_for_interp = mask_for_interp.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif mask_for_interp.dim() == 3:
                    mask_for_interp = mask_for_interp.unsqueeze(0)  # [1, C, H, W]
                
                text_mask_resized = F.interpolate(
                    mask_for_interp, 
                    size=orig_image.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                text_color_prob = self.analyze_text_color_from_mask(orig_image, text_mask_resized)
            else:
                # No text detected = not white
                text_color_prob = torch.tensor(0.0, device=c5_features.device)
            
            text_color_outputs.append(text_color_prob.unsqueeze(0))
            
            # Spatial knob size comparison
            knob_size_good = self.spatial_knob_size_comparison(
                plus_boxes, minus_boxes, plus_masks, minus_masks, hole_boxes, c5_features.device
            )
            knob_size_outputs.append(knob_size_good.unsqueeze(0))
            
            # Rule-based overall quality
            overall_quality = self.compute_overall_quality(
                knob_size_good, text_color_prob, len(hole_boxes), c5_features.device
            )
            overall_quality_outputs.append(overall_quality.unsqueeze(0))
        
        # Stack outputs
        perspective_out = torch.stack(perspective_outputs)
        text_color_out = torch.stack(text_color_outputs)
        knob_size_out = torch.stack(knob_size_outputs)
        overall_quality_out = torch.stack(overall_quality_outputs)
        
        return {
            'maskrcnn': maskrcnn_out,
            'perspective': perspective_out,
            'overall_quality': overall_quality_out,
            'text_color': text_color_out,
            'knob_size': knob_size_out
        } 