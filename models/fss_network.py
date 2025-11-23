import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import FeatureExtractor, SFEM, FeatureMatching, HPKIM


class FSSNetwork(nn.Module):
    """
    FSS Network (Meta-learning Support Segmentation) for Few-Shot Semantic Segmentation
    
    Components:
    1. Feature Extraction Module: Shared ResNet-50 backbone + PSPNet
    2. SFEM Module: Support Feature Enhancement with learnable ACM
    3. Feature Matching Module: Cosine similarity based feature matching
    4. HPKIM Module: Hierarchical Prior Knowledge Integration
    """
    def __init__(self, pretrained=True):
        super(FSSNetwork, self).__init__()
        
        # 1. Feature Extraction Module (shared for query and support)
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        
        # 2. SFEM modules for different feature levels
        self.sfem_low = SFEM(channels=256)   # For low-level features
        self.sfem_mid = SFEM(channels=1024)  # For mid-level features
        self.sfem_high = SFEM(channels=512)  # For high-level features
        
        # 3. Feature Matching modules
        self.matching_low = FeatureMatching(channels=256)
        self.matching_mid = FeatureMatching(channels=1024)
        self.matching_high = FeatureMatching(channels=512)
        
        # 4. HPKIM module
        self.hpkim = HPKIM(low_channels=256, mid_channels=1024, high_channels=512)
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # Binary segmentation (background/foreground)
        )
        
    def forward(self, query_img, support_img, support_mask):
        """
        Args:
            query_img: Query image tensor [B, 3, H, W]
            support_img: Support image tensor [B, 3, H, W]
            support_mask: Support mask tensor [B, H, W]
        
        Returns:
            pred: Segmentation prediction [B, 2, H, W]
        """
        b, _, h, w = query_img.shape
        
        # 1. Feature Extraction
        query_feats = self.feature_extractor(query_img)
        support_feats = self.feature_extractor(support_img)
        
        # Resize support mask to match feature sizes
        support_mask_low = F.interpolate(
            support_mask.unsqueeze(1).float(), 
            size=query_feats['low'].shape[-2:],
            mode='nearest',
            align_corners=None
        ).squeeze(1)
        
        support_mask_mid = F.interpolate(
            support_mask.unsqueeze(1).float(), 
            size=query_feats['mid'].shape[-2:],
            mode='nearest',
            align_corners=None
        ).squeeze(1)
        
        support_mask_high = F.interpolate(
            support_mask.unsqueeze(1).float(), 
            size=query_feats['high'].shape[-2:],
            mode='nearest',
            align_corners=None
        ).squeeze(1)
        
        # 2. SFEM - Support Feature Enhancement
        query_low_enh, support_low_enh = self.sfem_low(
            query_feats['low'], support_feats['low'], support_mask_low
        )
        query_mid_enh, support_mid_enh = self.sfem_mid(
            query_feats['mid'], support_feats['mid'], support_mask_mid
        )
        query_high_enh, support_high_enh = self.sfem_high(
            query_feats['high'], support_feats['high'], support_mask_high
        )
        
        # 3. Feature Matching
        matched_low, sim_low = self.matching_low(
            query_low_enh, support_low_enh, support_mask_low
        )
        matched_mid, sim_mid = self.matching_mid(
            query_mid_enh, support_mid_enh, support_mask_mid
        )
        matched_high, sim_high = self.matching_high(
            query_high_enh, support_high_enh, support_mask_high
        )
        
        # Prepare enhanced features for HPKIM
        query_feats_enh = {
            'low': matched_low,
            'mid': matched_mid,
            'high': matched_high
        }
        
        support_feats_enh = {
            'low': support_low_enh,
            'mid': support_mid_enh,
            'high': support_high_enh
        }
        
        # 4. HPKIM - Hierarchical Prior Knowledge Integration
        integrated_feat = self.hpkim(query_feats_enh, support_feats_enh, (h, w))
        
        # Final segmentation
        pred = self.segmentation_head(integrated_feat)
        
        return pred
    
    def predict(self, query_img, support_img, support_mask):
        """
        Prediction function for inference
        
        Args:
            query_img: Query image tensor [B, 3, H, W]
            support_img: Support image tensor [B, 3, H, W]
            support_mask: Support mask tensor [B, H, W]
        
        Returns:
            pred_mask: Predicted mask [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(query_img, support_img, support_mask)
            pred_mask = torch.argmax(pred, dim=1)
        return pred_mask
