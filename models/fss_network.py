"""
FSS Network - Few-Shot Segmentation Network
Main architecture combining all modules:
1. Feature Extraction (ResNet50 + PSPNet)
2. SFEM (Sparse Feature Enhancement Module)
3. Feature Matching Module
4. HPKIM (High-level Prior Knowledge Integration Module)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet50Backbone
from .sfem import SFEM
from .feature_matching import FeatureMatchingModule
from .hpkim import HPKIM


class FSSNetwork(nn.Module):
    """
    FSS Network for wild plant disease and pest segmentation
    Designed to reduce annotation cost in few-shot semantic segmentation
    """
    
    def __init__(self, config):
        super(FSSNetwork, self).__init__()
        
        self.config = config
        
        # 1. Feature Extraction: ResNet50 + PSPNet
        self.backbone = ResNet50Backbone(
            pretrained=config['model']['pretrained'],
            freeze=config['model']['freeze_backbone']
        )
        
        # 2. SFEM: Sparse Feature Enhancement Module
        self.sfem_query = SFEM(
            in_channels=512,  # PSPNet output channels
            out_channels=512,
            use_acm=config['model']['sfem']['use_acm'],
            learnable_correction=config['model']['sfem']['learnable_correction'],
            correction_init=config['model']['sfem']['correction_init']
        )
        
        self.sfem_support = SFEM(
            in_channels=512,
            out_channels=512,
            use_acm=config['model']['sfem']['use_acm'],
            learnable_correction=config['model']['sfem']['learnable_correction'],
            correction_init=config['model']['sfem']['correction_init']
        )
        
        # 3. Feature Matching Module
        self.feature_matching = FeatureMatchingModule(
            feature_dim=512,
            temperature=config['model']['matching']['temperature']
        )
        
        # 4. HPKIM: High-level Prior Knowledge Integration Module
        self.hpkim = HPKIM(
            low_level_channels=256,   # layer1 output
            high_level_channels=1024, # layer3 output
            output_channels=256,
            use_prior_knowledge=config['model']['hpkim']['use_prior_knowledge']
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)  # Binary segmentation (foreground/background)
        )
        
    def forward(self, query_img, support_img, support_mask):
        """
        Forward pass of FSS Network
        
        Args:
            query_img: Query image [B, 3, H, W]
            support_img: Support image [B, 3, H, W]
            support_mask: Support mask [B, 1, H, W]
            
        Returns:
            segmentation: Predicted segmentation [B, 2, H, W]
        """
        batch_size = query_img.shape[0]
        img_size = query_img.shape[-2:]
        
        # 1. Feature Extraction
        query_features = self.backbone(query_img)
        support_features = self.backbone(support_img)
        
        # Extract features at different levels
        query_psp = query_features['psp']      # [B, 512, H/32, W/32]
        support_psp = support_features['psp']  # [B, 512, H/32, W/32]
        
        query_low = query_features['layer1']   # [B, 256, H/4, W/4]
        query_high = query_features['layer3']  # [B, 1024, H/16, W/16]
        
        # 2. SFEM: Enhance features with attention on foreground
        query_enhanced, query_fg_mask = self.sfem_query(query_psp)
        support_enhanced, support_fg_mask = self.sfem_support(support_psp)
        
        # 3. Feature Matching: Compute similarity between query and support
        similarity_map = self.feature_matching(
            query_enhanced, 
            support_enhanced, 
            support_mask
        )
        
        # 4. HPKIM: Integrate prior knowledge and fuse features
        fused_features = self.hpkim(
            query_low,
            query_high,
            similarity_map
        )
        
        # 5. Final segmentation
        segmentation = self.segmentation_head(fused_features)
        
        # Upsample to original image size
        segmentation = F.interpolate(
            segmentation,
            size=img_size,
            mode='bilinear',
            align_corners=True
        )
        
        return segmentation
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_fss_network(config):
    """
    Build FSS Network from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: FSS Network model
    """
    model = FSSNetwork(config)
    return model
