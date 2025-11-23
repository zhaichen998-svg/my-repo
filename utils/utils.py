"""
Utility functions for FSS Network
"""

import os
import yaml
import torch
import numpy as np
from torch.nn import functional as F


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, loss


def compute_iou(pred, target, num_classes=2):
    """
    Compute Intersection over Union (IoU)
    
    Args:
        pred: Predicted segmentation [B, H, W]
        target: Ground truth segmentation [B, H, W]
        num_classes: Number of classes
        
    Returns:
        iou: Mean IoU across classes
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            # Both prediction and target are empty (no pixels of this class)
            # Perfect match in this case, so return IoU of 1.0
            iou = 1.0
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return np.mean(ious)


def compute_dice(pred, target):
    """
    Compute Dice coefficient
    
    Args:
        pred: Predicted segmentation [B, H, W]
        target: Ground truth segmentation [B, H, W]
        
    Returns:
        dice: Dice coefficient
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum().float()
    union = pred.sum().float() + target.sum().float()
    
    if union == 0:
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice.item()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, config):
    """
    Get optimizer based on configuration
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        optimizer: Optimizer instance
    """
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        
    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler_name = config['training']['lr_scheduler'].lower()
    
    if scheduler_name == 'step':
        step_size = config['training']['lr_decay_step']
        gamma = config['training']['lr_decay_gamma']
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name == 'cosine':
        T_max = config['training']['epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max
        )
    else:
        scheduler = None
    
    return scheduler
