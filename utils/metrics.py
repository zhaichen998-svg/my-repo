import torch
import numpy as np


def compute_iou(pred, target, num_classes=2):
    """
    Compute Intersection over Union (IoU) for semantic segmentation
    
    Args:
        pred: Predicted mask [B, H, W] or [B, C, H, W]
        target: Ground truth mask [B, H, W]
        num_classes: Number of classes
    
    Returns:
        iou: Mean IoU across all classes
    """
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
    
    return np.mean(ious) if ious else 0.0


def compute_dice(pred, target, num_classes=2):
    """
    Compute Dice coefficient for semantic segmentation
    
    Args:
        pred: Predicted mask [B, H, W] or [B, C, H, W]
        target: Ground truth mask [B, H, W]
        num_classes: Number of classes
    
    Returns:
        dice: Mean Dice coefficient across all classes
    """
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.view(-1)
    target = target.view(-1)
    
    dices = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
            dices.append(dice.item())
    
    return np.mean(dices) if dices else 0.0


def compute_pixel_accuracy(pred, target):
    """
    Compute pixel-wise accuracy
    
    Args:
        pred: Predicted mask [B, H, W] or [B, C, H, W]
        target: Ground truth mask [B, H, W]
    
    Returns:
        accuracy: Pixel accuracy
    """
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).sum().float()
    total = target.numel()
    
    return (correct / total).item()
