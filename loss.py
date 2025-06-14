import torch
import torch.nn as nn
import torch.nn.functional as F

def custom_loss(outputs, targets):
    """
    Combined loss function for CustomMaskRCNN with rule-based overall quality
    
    Args:
        outputs: Model outputs dict with keys:
            - 'maskrcnn': Either loss dict (training) or predictions list (inference)
            - 'perspective': [B, 8] perspective points
            - 'overall_quality': [B] rule-based quality (not trained directly)
            - 'text_color': [B] text color probabilities (0-1)
            - 'knob_size': [B] knob size assessment (0-1)
        targets: Target dict with keys:
            - 'perspective': [B, 8] normalized perspective points
            - 'overall_quality': [B] quality labels (0,1,2) - used for validation only
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

    # Text color (binary BCE with logits)
    # Note: outputs['text_color'] now contains probabilities (0-1), not logits
    pred_text = outputs['text_color'].squeeze(-1) if outputs['text_color'].dim() > 1 else outputs['text_color']
    target_text = targets['text_color'].float()
    
    # Convert probabilities to logits for BCE with logits
    pred_text_logits = torch.log(pred_text + 1e-8) - torch.log(1 - pred_text + 1e-8)
    loss_text = F.binary_cross_entropy_with_logits(pred_text_logits, target_text)

    # Knob size (binary BCE with logits)
    # Note: outputs['knob_size'] now contains probabilities (0-1), not logits
    pred_knob = outputs['knob_size'].squeeze(-1) if outputs['knob_size'].dim() > 1 else outputs['knob_size']
    target_knob = targets['knob_size'].float()
    
    # Convert probabilities to logits for BCE with logits
    pred_knob_logits = torch.log(pred_knob + 1e-8) - torch.log(1 - pred_knob + 1e-8)
    loss_knob = F.binary_cross_entropy_with_logits(pred_knob_logits, target_knob)

    # Overall quality is now rule-based, so we don't train it directly
    # But we can compute a consistency loss to encourage the rule-based logic
    # to align with ground truth when available
    pred_quality = outputs['overall_quality']  # [B] - rule-based predictions
    target_quality = targets['overall_quality'].long()  # [B] - ground truth
    
    # Convert rule-based binary predictions to match 3-class targets
    # Rule-based: 0=good, 1=bad
    # Ground truth: 0=good, 1=bad, 2=unknown
    # We'll only penalize when ground truth is 0 or 1 (not unknown)
    valid_targets = target_quality < 2  # Ignore unknown samples
    
    if valid_targets.any():
        # Convert rule-based predictions to match ground truth format
        rule_based_quality = pred_quality[valid_targets].squeeze().long()  # Remove extra dimensions
        gt_quality = target_quality[valid_targets]
        
        # Ensure we have at least 1D tensors
        if rule_based_quality.dim() == 0:
            rule_based_quality = rule_based_quality.unsqueeze(0)
        if gt_quality.dim() == 0:
            gt_quality = gt_quality.unsqueeze(0)
        
        # Simple binary cross-entropy for consistency
        # Create logits for 2-class problem (good vs bad)
        rule_based_probs = rule_based_quality.float()
        logits = torch.stack([1 - rule_based_probs, rule_based_probs], dim=1)
        loss_quality_consistency = F.cross_entropy(logits, gt_quality)
    else:
        loss_quality_consistency = torch.tensor(0.0, device=outputs['perspective'].device)

    # Weighted total loss
    # Reduced weight for quality consistency since it's rule-based
    total_loss = (
        loss_maskrcnn + 
        0.8 * loss_persp +  # Perspective points
        0.3 * loss_quality_consistency +  # Light consistency loss
        1.2 * loss_text +  # Text color (important)
        1.2 * loss_knob   # Knob size (important)
    )
    
    return {
        'total_loss': total_loss,
        'maskrcnn_loss': loss_maskrcnn,
        'perspective_loss': loss_persp,
        'quality_consistency_loss': loss_quality_consistency,
        'text_color_loss': loss_text,
        'knob_size_loss': loss_knob
    } 