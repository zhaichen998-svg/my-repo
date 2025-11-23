"""
HPKIM - High-level Prior Knowledge Integration Module
Provides prior knowledge features as additional cues for feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPKIM(nn.Module):
    """
    High-level Prior Knowledge Integration Module
    Combines coarse-grained and fine-grained features with prior knowledge
    to guide the segmentation process
    """
    
    def __init__(self, low_level_channels=256, high_level_channels=512, 
                 output_channels=256, use_prior_knowledge=True):
        super(HPKIM, self).__init__()
        
        self.use_prior_knowledge = use_prior_knowledge
        
        # Low-level feature processing (fine-grained)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # High-level feature processing (coarse-grained)
        self.high_level_conv = nn.Sequential(
            nn.Conv2d(high_level_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Prior knowledge extraction
        if self.use_prior_knowledge:
            self.prior_conv = nn.Sequential(
                nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        
        # Feature fusion
        fusion_input_channels = output_channels * 3 if use_prior_knowledge else output_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_input_channels, output_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention for weighted fusion
        self.attention = nn.Sequential(
            nn.Conv2d(output_channels, output_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels // 4, 3 if use_prior_knowledge else 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, low_level_features, high_level_features, similarity_map):
        """
        Args:
            low_level_features: Fine-grained features from backbone (e.g., layer1) [B, C_low, H, W]
            high_level_features: Coarse-grained features from backbone (e.g., layer3) [B, C_high, H', W']
            similarity_map: Similarity map from feature matching [B, 1, H'', W'']
        Returns:
            fused_features: Fused features with prior knowledge [B, C_out, H, W]
        """
        # Get target size from low-level features
        target_size = low_level_features.shape[-2:]
        
        # Process low-level features
        low_features = self.low_level_conv(low_level_features)
        
        # Process and upsample high-level features
        high_features = self.high_level_conv(high_level_features)
        high_features = F.interpolate(high_features, size=target_size, 
                                     mode='bilinear', align_corners=True)
        
        # Upsample similarity map
        similarity_upsampled = F.interpolate(similarity_map, size=target_size,
                                            mode='bilinear', align_corners=True)
        
        # Extract prior knowledge if enabled
        if self.use_prior_knowledge:
            # Combine high-level and similarity for prior knowledge
            prior_input = torch.cat([high_features, 
                                    similarity_upsampled.expand(-1, high_features.shape[1], -1, -1)], 
                                   dim=1)
            prior_features = self.prior_conv(prior_input)
            
            # Concatenate all features
            combined = torch.cat([low_features, high_features, prior_features], dim=1)
            
            # Compute attention weights
            attention_input = torch.cat([low_features, high_features, prior_features], dim=1)
            attention_input = self.fusion(attention_input)
            attention_weights = self.attention(attention_input)
            
            # Weighted fusion
            weighted_low = low_features * attention_weights[:, 0:1, :, :]
            weighted_high = high_features * attention_weights[:, 1:2, :, :]
            weighted_prior = prior_features * attention_weights[:, 2:3, :, :]
            
            fused = weighted_low + weighted_high + weighted_prior
        else:
            # Simple fusion without prior knowledge
            combined = torch.cat([low_features, high_features], dim=1)
            fused = self.fusion(combined)
        
        return fused
