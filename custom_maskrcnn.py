import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # num_classes: 4 masks + background
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Custom heads use global average pooled C5 features ([B, 2048])
        custom_in_dim = 2048
        self.perspective_head = nn.Sequential(
            nn.Linear(custom_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)
        )
        self.quality_head = nn.Sequential(
            nn.Linear(custom_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        self.text_color_head = nn.Sequential(
            nn.Linear(custom_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.knob_size_head = nn.Sequential(
            nn.Linear(custom_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, images, targets=None):
        # Standard Mask R-CNN outputs
        out = self.model(images, targets) if self.training else self.model(images)
        # Get C5 features from backbone.body and global average pool
        if isinstance(images, list):
            imgs = torch.stack(images)
        else:
            imgs = images
        backbone_out = self.model.backbone.body(imgs)  # OrderedDict
        c5 = backbone_out['3']  # [B, 2048, H/32, W/32] - the deepest feature level
        gap = torch.mean(c5, dim=[2, 3])  # shape [B, 2048]
        perspective = self.perspective_head(gap)
        quality = self.quality_head(gap)
        text_color = self.text_color_head(gap)
        knob_size = self.knob_size_head(gap)
        out_dict = {
            'maskrcnn': out,
            'perspective': perspective,
            'overall_quality': quality,
            'text_color': text_color,
            'knob_size': knob_size
        }
        return out_dict 