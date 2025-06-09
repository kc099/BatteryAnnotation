import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4

class HierarchicalQualityModel(nn.Module):
    """Model with separate quality assessment heads for each component"""
    
    def __init__(self, num_seg_classes=6, num_features=12):
        super().__init__()
        
        # Backbone - EfficientNet for better efficiency
        self.backbone = efficientnet_b4(pretrained=True)
        backbone_out_channels = 1792
        
        # Remove the classifier head
        self.backbone.classifier = nn.Identity()
        
        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(backbone_out_channels, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_seg_classes, 1)
        )
        
        # Feature processor combining CNN and hand-crafted features
        self.feature_processor = nn.Sequential(
            nn.Linear(backbone_out_channels + num_features, 512),
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
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for focusing on important regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(backbone_out_channels, backbone_out_channels // 8, 1),
            nn.BatchNorm2d(backbone_out_channels // 8),
            nn.ReLU(),
            nn.Conv2d(backbone_out_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def _make_quality_head(self, in_features, component_name):
        """Create a quality assessment head for a specific component"""
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, features=None):
        # Extract features using backbone
        # EfficientNet returns features at different scales
        backbone_features = self.backbone.features(x)
        
        # Apply spatial attention
        attention_map = self.spatial_attention(backbone_features)
        attended_features = backbone_features * attention_map
        
        # Global average pooling for classification
        global_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        # Get backbone output (already includes global pooling)
        backbone_out = F.adaptive_avg_pool2d(backbone_features, 1).flatten(1)
        
        # Segmentation output
        seg_output = self.seg_decoder(backbone_features)
        
        # Combine CNN features with hand-crafted features
        if features is not None:
            combined_features = torch.cat([backbone_out, features], dim=1)
        else:
            # Create dummy features if none provided
            dummy_features = torch.zeros(backbone_out.shape[0], 12, device=backbone_out.device)
            combined_features = torch.cat([backbone_out, dummy_features], dim=1)
        
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
            'attention_map': attention_map,
            'features': processed_features
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
        
        # Quality losses (binary classification)
        self.quality_criterion = nn.BCELoss(reduction='none')
    
    def forward(self, predictions, targets):
        losses = {}
        
        # Segmentation loss (multi-channel binary)
        seg_loss = 0
        for c in range(predictions['segmentation'].shape[1]):
            channel_loss = self.seg_criterion(
                predictions['segmentation'][:, c], 
                targets['masks'][:, c]
            )
            seg_loss += channel_loss.mean()
        losses['seg_loss'] = seg_loss / predictions['segmentation'].shape[1]
        
        # Component quality losses (only for labeled samples)
        component_losses = []
        
        for component in ['hole', 'text', 'knob', 'surface']:
            pred_key = f'{component}_quality'
            target_key = f'{component}_quality'
            
            # Get predictions and targets
            preds = predictions[pred_key].squeeze()
            targets_comp = targets[target_key]
            
            # Create mask for valid labels (not -1)
            valid_mask = targets_comp >= 0
            
            if valid_mask.sum() > 0:
                # Calculate loss only for valid labels
                valid_preds = preds[valid_mask]
                valid_targets = targets_comp[valid_mask]
                valid_weights = targets['weight'][valid_mask]
                
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
        overall_preds = predictions['overall_quality'].squeeze()
        overall_targets = targets['overall_quality']
        overall_valid_mask = overall_targets >= 0
        
        if overall_valid_mask.sum() > 0:
            valid_overall_preds = overall_preds[overall_valid_mask]
            valid_overall_targets = overall_targets[overall_valid_mask]
            valid_overall_weights = targets['weight'][overall_valid_mask]
            
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