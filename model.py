import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

class HierarchicalQualityModel(nn.Module):
    """DeepLabV3+ based model with segmentation and classification heads"""
    
    def __init__(self, num_seg_classes=6, num_features=12, input_size=512):
        super().__init__()
        
        # DeepLabV3+ backbone with ResNet50
        self.backbone = deeplabv3_resnet50(pretrained=True, progress=True)
        
        # Keep the original classifier for ASPP features
        # We'll just use the ASPP output, not the final classification
        
        # Get backbone feature dimensions
        backbone_out_channels = 2048  # ResNet50 final layer
        
        # Custom segmentation head (replace the original)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # DeepLabV3+ outputs 256 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_seg_classes, 1)
        )
        
        # Global feature extractor for classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature processor combining CNN and hand-crafted features
        # Use 256 channels from DeepLabV3+ output
        self.feature_processor = nn.Sequential(
            nn.Linear(256 + num_features, 512),  # 256 from global pooled features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Component-specific quality heads
        self.hole_quality_head = self._make_quality_head(256, "holes")
        self.text_quality_head = self._make_quality_head(256, "text")
        self.knob_quality_head = self._make_quality_head(256, "knobs")
        self.surface_quality_head = self._make_quality_head(256, "surface")
        
        # Overall quality head (takes all component predictions as additional input)
        self.overall_quality_head = nn.Sequential(
            nn.Linear(256 + 4, 128),  # +4 for component quality predictions
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # No sigmoid - BCEWithLogitsLoss expects logits
        )
        
        # Store input size for proper resizing
        self.input_size = input_size
    
    def _make_quality_head(self, in_features, component_name):
        """Create a quality assessment head for a specific component"""
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # No sigmoid - BCEWithLogitsLoss expects logits
        )
    
    def forward(self, x, features=None):
        batch_size = x.shape[0]
        original_size = x.shape[-2:]
        
        # Use DeepLabV3+ features directly
        # Get backbone features
        backbone_features = self.backbone.backbone(x)
        x_feat = backbone_features['out']  # High-level features from ResNet50
        
        # Get ASPP features using the pre-trained ASPP module
        aspp_features = self.backbone.classifier[0](x_feat)  # ASPP output (256 channels)
        
        # Custom segmentation head (replaces the final 1x1 conv of DeepLabV3+)
        seg_output = self.seg_head(aspp_features)
        
        # Upsample segmentation to input size
        seg_output = F.interpolate(seg_output, size=original_size, 
                                 mode='bilinear', align_corners=False)
        
        # Classification branch - global pooling from ASPP features
        global_features = self.global_pool(aspp_features).flatten(1)
        
        # Combine CNN features with hand-crafted features
        if features is not None:
            combined_features = torch.cat([global_features, features], dim=1)
        else:
            # Create dummy features if none provided
            dummy_features = torch.zeros(batch_size, 12, device=global_features.device)
            combined_features = torch.cat([global_features, dummy_features], dim=1)
        
        # Process combined features
        processed_features = self.feature_processor(combined_features)
        
        # Component quality predictions
        hole_quality = self.hole_quality_head(processed_features)
        text_quality = self.text_quality_head(processed_features)
        knob_quality = self.knob_quality_head(processed_features)
        surface_quality = self.surface_quality_head(processed_features)
        
        # Overall quality (considering component qualities)
        component_qualities = torch.cat([
            hole_quality, text_quality, knob_quality, surface_quality
        ], dim=1)
        
        overall_input = torch.cat([processed_features, component_qualities], dim=1)
        overall_quality = self.overall_quality_head(overall_input)
        
        return {
            'segmentation': seg_output,
            'hole_quality': hole_quality,
            'text_quality': text_quality,
            'knob_quality': knob_quality,
            'surface_quality': surface_quality,
            'overall_quality': overall_quality,
            'component_qualities': component_qualities,
            'features': processed_features,
            'seg_features': aspp_features  # For visualization
        }

class ComponentAwareLoss(nn.Module):
    """Loss function that handles component-wise quality assessment"""
    
    def __init__(self, seg_weight=1.0, component_weight=1.5, overall_weight=2.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.component_weight = component_weight
        self.overall_weight = overall_weight
        
        # Segmentation loss
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Quality losses (binary classification) - Use BCEWithLogitsLoss for mixed precision
        self.quality_criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets):
        losses = {}
        
        # Segmentation loss (multi-channel binary)
        seg_loss = 0
        seg_pred = predictions['segmentation']
        seg_target = targets['masks']
        
        # Ensure spatial dimensions match
        if seg_pred.shape[-2:] != seg_target.shape[-2:]:
            seg_target = F.interpolate(seg_target, size=seg_pred.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        
        for c in range(seg_pred.shape[1]):
            channel_loss = self.seg_criterion(
                seg_pred[:, c], 
                seg_target[:, c]
            )
            seg_loss += channel_loss.mean()
        losses['seg_loss'] = seg_loss / seg_pred.shape[1]
        
        # Component quality losses (only for labeled samples)
        component_losses = []
        
        for component in ['hole', 'text', 'knob', 'surface']:
            pred_key = f'{component}_quality'
            target_key = f'{component}_quality'
            
            # Get predictions and targets
            preds = predictions[pred_key].flatten()  # Use flatten instead of squeeze
            targets_comp = targets[target_key].flatten()  # Ensure same shape
            
            # Create mask for valid labels (not -1)
            valid_mask = targets_comp >= 0
            
            if valid_mask.sum() > 0:
                # Calculate loss only for valid labels
                valid_preds = preds[valid_mask]
                valid_targets = targets_comp[valid_mask]
                valid_weights = targets['weight'].flatten()[valid_mask]
                
                loss = self.quality_criterion(valid_preds, valid_targets)
                weighted_loss = (loss * valid_weights).mean()
                
                component_losses.append(weighted_loss)
                losses[f'{component}_loss'] = weighted_loss
        
        # Average component loss
        if component_losses:
            losses['component_loss'] = torch.stack(component_losses).mean()
        else:
            losses['component_loss'] = torch.tensor(0.0, device=predictions['hole_quality'].device)
        
        # Overall quality loss
        overall_preds = predictions['overall_quality'].flatten()  # Use flatten instead of squeeze
        overall_targets = targets['overall_quality'].flatten()  # Ensure same shape
        overall_valid_mask = overall_targets >= 0
        
        if overall_valid_mask.sum() > 0:
            valid_overall_preds = overall_preds[overall_valid_mask]
            valid_overall_targets = overall_targets[overall_valid_mask]
            valid_overall_weights = targets['weight'].flatten()[overall_valid_mask]
            
            overall_loss = self.quality_criterion(valid_overall_preds, valid_overall_targets)
            losses['overall_loss'] = (overall_loss * valid_overall_weights).mean()
        else:
            losses['overall_loss'] = torch.tensor(0.0, device=overall_preds.device)
        
        # Total loss
        losses['total_loss'] = (
            self.seg_weight * losses['seg_loss'] +
            self.component_weight * losses['component_loss'] +
            self.overall_weight * losses['overall_loss']
        )
        
        return losses 