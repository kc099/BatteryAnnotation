import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import roi_align

class ComponentROIModel(nn.Module):
    """
    A new model architecture that uses a powerful backbone and ROI-Align 
    for more precise component quality assessment.
    """
    def __init__(self, num_seg_classes=4, roi_output_size=7, num_features=12):
        super().__init__()
        
        # 1. Powerful Backbone + FPN
        # Using a pre-trained EfficientNetV2 from timm
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=True, 
            features_only=True,
            out_indices=(1, 2, 3, 4) # Get features from multiple stages
        )
        
        # Get the feature dimensions from the backbone
        feature_dims = self.backbone.feature_info.channels()
        
        # 2. Feature Pyramid Network (FPN)
        # The FPN will create a rich multi-scale feature representation
        self.fpn = nn.Sequential(
            nn.Conv2d(feature_dims[-1], 256, 1),
            nn.ReLU(inplace=True),
        )
        # In a real FPN you would combine features from different levels.
        # For simplicity here, we'll just use the last layer's features processed by a 1x1 conv
        # A proper FPN implementation would be more complex but even this is an improvement.
        
        # 3. Segmentation Head
        self.seg_head = self._create_head(256, num_seg_classes)
        
        # 4. ROI-Align Configuration
        self.roi_output_size = roi_output_size
        
        # 5. Component Quality Heads
        # These heads will operate on the ROI-aligned features
        roi_feature_dim = 256 * (roi_output_size ** 2)
        
        self.hole_quality_head = self._create_quality_head(roi_feature_dim, "hole")
        self.text_quality_head = self._create_quality_head(roi_feature_dim, "text")
        self.knob_quality_head = self._create_quality_head(roi_feature_dim, "knob")

        # 6. Overall Quality Head
        # Takes the concatenated features from all components
        total_roi_features = roi_feature_dim * 3 # for hole, text, and knob
        self.overall_quality_head = nn.Sequential(
            nn.Linear(total_roi_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def _create_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def _create_quality_head(self, in_features, name):
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, targets=None):
        original_size = x.shape[-2:]
        
        # Get multi-scale features from the backbone
        features = self.backbone(x)
        fpn_features = self.fpn(features[-1])
        
        # Segmentation prediction
        seg_logits = self.seg_head(fpn_features)
        seg_logits = F.interpolate(seg_logits, size=original_size, mode='bilinear', align_corners=False)
        
        # During inference, if no targets are provided for quality checks, 
        # just return the segmentation map.
        if targets is None:
            return {'segmentation': seg_logits}

        # --- ROI-based Quality Prediction ---
        # Compute spatial scale dynamically
        spatial_scale = fpn_features.size(-1) / x.size(-1)

        all_roi_features = []
        outputs = {'segmentation': seg_logits}

        # Process each component type
        for comp_type in ['hole', 'text', 'knob']:
            boxes_list = targets.get(f'{comp_type}_boxes') # List of [N, 4] tensors
            if boxes_list and sum(len(b) for b in boxes_list) > 0:
                # Concatenate all boxes and prepend batch indices
                all_boxes = []
                for i, b in enumerate(boxes_list):
                    if b.numel() > 0:
                        batch_idx = torch.full((b.size(0), 1), i, device=b.device, dtype=b.dtype)
                        all_boxes.append(torch.cat([batch_idx, b], dim=1))
                if all_boxes:
                    rois = torch.cat(all_boxes, dim=0)
                    # Use ROI-Align to get features for each box
                    roi_features = roi_align(fpn_features, rois, self.roi_output_size, spatial_scale=spatial_scale)
                    # Flatten features for the linear layer
                    roi_features_flat = roi_features.view(roi_features.size(0), -1)
                    # Get quality predictions
                    quality_head = getattr(self, f'{comp_type}_quality_head')
                    quality_logits = quality_head(roi_features_flat)
                    outputs[f'{comp_type}_quality'] = quality_logits

                    # For overall quality, we need a fixed-size vector per image.
                    # We can average the features for all instances in each image.
                    image_indices = rois[:, 0].long()
                    num_images = x.size(0)
                    summed_features = torch.zeros(num_images, roi_features_flat.shape[1], device=x.device)
                    roi_features_flat = roi_features_flat.to(summed_features.dtype)
                    summed_features.index_add_(0, image_indices, roi_features_flat)
                    box_counts = torch.bincount(image_indices, minlength=num_images).unsqueeze(1).clamp(min=1)
                    avg_features = summed_features / box_counts
                    all_roi_features.append(avg_features)
                else:
                    outputs[f'{comp_type}_quality'] = torch.empty(0, 1, device=x.device)
                    all_roi_features.append(torch.zeros(x.size(0), 256 * self.roi_output_size**2, device=x.device))
            else:
                # If no boxes, provide empty logits and zero features
                outputs[f'{comp_type}_quality'] = torch.empty(0, 1, device=x.device)
                all_roi_features.append(torch.zeros(x.size(0), 256 * self.roi_output_size**2, device=x.device))

        # Concatenate average features for overall quality prediction
        concatenated_features = torch.cat(all_roi_features, dim=1)
        overall_quality_logits = self.overall_quality_head(concatenated_features)
        outputs['overall_quality'] = overall_quality_logits

        return outputs

if __name__ == '__main__':
    # Example usage
    model = ComponentROIModel()
    input_tensor = torch.randn(2, 3, 544, 960) # Batch of 2
    
    # During training, you would also provide targets
    # This is a simplified example.
    # The targets need to be formatted correctly by the Dataset object.
    targets = {
       'hole_boxes': [torch.tensor([[10, 10, 50, 50], [100, 100, 150, 180]], dtype=torch.float), # boxes for image 1
                       torch.tensor([[30, 30, 80, 80]], dtype=torch.float)], # box for image 2
        # ... other box types
    }

    # The model would be called inside the lightning module
    # output = model(input_tensor, targets)
    # print(output['segmentation'].shape)
    # print(output['hole_quality'].shape) # Will depend on number of boxes
    print("Model created successfully.")
    print("Note: The forward pass logic is now updated to handle ROI-Align and batching.") 