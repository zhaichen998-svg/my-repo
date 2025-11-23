"""
SFEM - Sparse Feature Enhancement Module
Integrates ACM (Attention Coordination Module) with learnable correction coefficients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCoordinationModule(nn.Module):
    """Attention Coordination Module (ACM) for focusing on foreground"""
    
    def __init__(self, in_channels, reduction=16):
        super(AttentionCoordinationModule, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa
        
        return x_sa


class SFEM(nn.Module):
    """
    Sparse Feature Enhancement Module
    Designed to focus on foreground and reduce irrelevant background information
    """
    
    def __init__(self, in_channels, out_channels=512, use_acm=True, 
                 learnable_correction=True, correction_init=1.0):
        super(SFEM, self).__init__()
        
        self.use_acm = use_acm
        self.learnable_correction = learnable_correction
        
        # ACM module
        if self.use_acm:
            self.acm = AttentionCoordinationModule(in_channels)
        
        # Learnable correction coefficient
        if self.learnable_correction:
            self.correction_coeff = nn.Parameter(
                torch.ones(1) * correction_init
            )
        else:
            self.correction_coeff = correction_init
        
        # Feature transformation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Background suppression
        self.bg_suppression = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply ACM if enabled
        if self.use_acm:
            x = self.acm(x)
        
        # Apply correction coefficient (learnable or fixed)
        x = x * self.correction_coeff
        
        # Feature transformation
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Background suppression
        fg_mask = self.bg_suppression(x)
        x_enhanced = x * fg_mask
        
        return x_enhanced, fg_mask
