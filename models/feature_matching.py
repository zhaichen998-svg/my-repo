"""
Feature Matching Module
Uses cosine similarity to match query and support features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchingModule(nn.Module):
    """
    Feature Matching Module using cosine similarity
    Matches enhanced query and support features from SFEM
    """
    
    def __init__(self, feature_dim=512, temperature=1.0):
        super(FeatureMatchingModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Feature projection for better matching
        self.query_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.support_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Similarity refinement
        self.similarity_refine = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def masked_average_pooling(self, features, mask):
        """
        Compute masked average pooling for support features
        Args:
            features: [B, C, H, W]
            mask: [B, 1, H, W]
        Returns:
            pooled: [B, C, 1, 1]
        """
        mask = F.interpolate(mask, size=features.shape[-2:], 
                            mode='bilinear', align_corners=True)
        
        # Masked pooling
        masked_features = features * mask
        pooled = masked_features.sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + 1e-5)
        
        return pooled
    
    def cosine_similarity(self, query_features, support_prototype):
        """
        Compute cosine similarity between query features and support prototype
        Args:
            query_features: [B, C, H, W]
            support_prototype: [B, C, 1, 1]
        Returns:
            similarity: [B, 1, H, W]
        """
        # Normalize features
        query_norm = F.normalize(query_features, p=2, dim=1)
        support_norm = F.normalize(support_prototype, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = (query_norm * support_norm).sum(dim=1, keepdim=True)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def forward(self, query_features, support_features, support_mask):
        """
        Args:
            query_features: Enhanced query features from SFEM [B, C, H, W]
            support_features: Enhanced support features from SFEM [B, C, H, W]
            support_mask: Support mask [B, 1, H, W]
        Returns:
            similarity_map: Similarity map [B, 1, H, W]
        """
        # Project features
        query_proj = self.query_proj(query_features)
        support_proj = self.support_proj(support_features)
        
        # Compute support prototype via masked average pooling
        support_prototype = self.masked_average_pooling(support_proj, support_mask)
        
        # Compute cosine similarity
        similarity_map = self.cosine_similarity(query_proj, support_prototype)
        
        # Refine similarity map
        similarity_refined = self.similarity_refine(similarity_map)
        
        return similarity_refined
