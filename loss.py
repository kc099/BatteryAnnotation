import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_loss(outputs, targets):
    """
    Combined loss function for CustomMaskRCNN
    
    Args:
        outputs: Model outputs dict with keys:
            - 'maskrcnn': Either loss dict (training) or predictions list (inference)
            - 'perspective': [B, 8] perspective points
            - 'overall_quality': [B, 3] quality logits
            - 'text_color': [B, 1] text color logits
            - 'knob_size': [B, 1] knob size logits
        targets: Target dict with keys:
            - 'perspective': [B, 8] normalized perspective points
            - 'overall_quality': [B] quality labels (0,1,2)
            - 'text_color': [B] binary labels
            - 'knob_size': [B] binary labels
    """
    
    # Mask R-CNN losses
    if isinstance(outputs['maskrcnn'], dict) and 'loss_classifier' in outputs['maskrcnn']:
        # Training mode: maskrcnn outputs are losses
        maskrcnn_losses = outputs['maskrcnn']
        loss_maskrcnn = sum(maskrcnn_losses.values())
    elif isinstance(outputs['maskrcnn'], list):
        # Inference mode: no maskrcnn loss
        loss_maskrcnn = torch.tensor(0.0, device=outputs['perspective'].device)
    else:
        # Fallback
        loss_maskrcnn = torch.tensor(0.0, device=outputs['perspective'].device)

    # Perspective points regression (MSE)
    pred_persp = outputs['perspective']
    target_persp = targets['perspective']
    loss_persp = F.mse_loss(pred_persp, target_persp)

    # Overall quality (3-class cross-entropy)
    pred_quality = outputs['overall_quality']
    target_quality = targets['overall_quality'].long()
    loss_quality = F.cross_entropy(pred_quality, target_quality)

    # Text color (binary BCE)
    pred_text = outputs['text_color'].squeeze(-1) if outputs['text_color'].dim() > 1 else outputs['text_color']
    target_text = targets['text_color'].float()
    loss_text = F.binary_cross_entropy_with_logits(pred_text, target_text)

    # Knob size (binary BCE)
    pred_knob = outputs['knob_size'].squeeze(-1) if outputs['knob_size'].dim() > 1 else outputs['knob_size']
    target_knob = targets['knob_size'].float()
    loss_knob = F.binary_cross_entropy_with_logits(pred_knob, target_knob)

    # Weighted total loss
    total_loss = (
        loss_maskrcnn + 
        0.1 * loss_persp +  # Lower weight for perspective
        loss_quality + 
        0.5 * loss_text + 
        0.5 * loss_knob
    )
    
    return {
        'total_loss': total_loss,
        'maskrcnn_loss': loss_maskrcnn,
        'perspective_loss': loss_persp,
        'quality_loss': loss_quality,
        'text_color_loss': loss_text,
        'knob_size_loss': loss_knob
    } 