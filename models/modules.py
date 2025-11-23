import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class PSPModule(nn.Module):
    """Pyramid Pooling Module for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, out_channels, size) for size in sizes
        ])
        self.bottleneck = nn.Conv2d(
            in_channels + len(sizes) * out_channels, out_channels, 
            kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        
    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)
    
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(
            stage(feats), size=(h, w), mode='bilinear', align_corners=True
        ) for stage in self.stages]
        priors.append(feats)
        out = self.relu(self.bottleneck(torch.cat(priors, 1)))
        return out


class FeatureExtractor(nn.Module):
    """Feature extraction module using ResNet-50 backbone and PSPNet"""
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=pretrained)
        
        # Extract layers from ResNet-50
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # Low-level features
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  # Mid-level features
        self.layer4 = resnet.layer4  # High-level features
        
        # PSPNet module for high-level features
        self.psp = PSPModule(2048, 512)
        
    def forward(self, x):
        # Extract features at different levels
        x = self.layer0(x)
        low_feat = self.layer1(x)      # 256 channels
        x = self.layer2(low_feat)      # 512 channels
        mid_feat = self.layer3(x)      # 1024 channels
        high_feat = self.layer4(mid_feat)  # 2048 channels
        
        # Apply PSPNet
        psp_feat = self.psp(high_feat)  # 512 channels
        
        return {
            'low': low_feat,    # Low-level features
            'mid': mid_feat,    # Mid-level features
            'high': psp_feat    # High-level PSP features
        }


class ACM(nn.Module):
    """Attention Calibration Module with learnable correction coefficient"""
    def __init__(self, channels):
        super(ACM, self).__init__()
        # Learnable correction coefficient
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attn(x)
        x_ca = x * ca
        
        # Spatial attention
        sa = self.spatial_attn(x_ca)
        x_sa = x_ca * sa
        
        # Apply learnable correction
        out = self.alpha * x_sa + self.beta
        return out


class SFEM(nn.Module):
    """Support Feature Enhancement Module with ACM"""
    def __init__(self, channels):
        super(SFEM, self).__init__()
        self.acm_query = ACM(channels)
        self.acm_support = ACM(channels)
        
        # Background filtering
        self.bg_filter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, query_feat, support_feat, support_mask):
        # Enhance query features
        query_enhanced = self.acm_query(query_feat)
        
        # Mask support features to filter background
        support_masked = support_feat * support_mask.unsqueeze(1)
        
        # Enhance support features
        support_enhanced = self.acm_support(support_masked)
        
        # Filter background noise
        support_enhanced = self.bg_filter(support_enhanced)
        
        return query_enhanced, support_enhanced


class FeatureMatching(nn.Module):
    """Feature matching module using cosine similarity"""
    def __init__(self, channels):
        super(FeatureMatching, self).__init__()
        self.channels = channels
        
    def masked_average_pooling(self, feat, mask):
        """Compute masked average pooling"""
        mask = F.interpolate(
            mask.unsqueeze(1).float(), 
            size=feat.shape[-2:], 
            mode='nearest',  # Use nearest for binary masks
            align_corners=None
        )
        feat_masked = feat * mask
        feat_avg = feat_masked.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)  # Larger epsilon for stability
        return feat_avg
    
    def forward(self, query_feat, support_feat, support_mask):
        b, c, h, w = query_feat.shape
        
        # Compute support prototype via masked average pooling
        support_proto = self.masked_average_pooling(support_feat, support_mask)
        support_proto = support_proto.view(b, c, 1, 1)
        
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_feat, p=2, dim=1)
        support_norm = F.normalize(support_proto, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = (query_norm * support_norm).sum(dim=1, keepdim=True)
        
        # Transfer knowledge from support to query
        matched_feat = query_feat * similarity
        
        return matched_feat, similarity


class HPKIM(nn.Module):
    """Hierarchical Prior Knowledge Integration Module"""
    def __init__(self, low_channels=256, mid_channels=1024, high_channels=512):
        super(HPKIM, self).__init__()
        
        # Feature pooling at different scales
        self.pool_coarse = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_fine = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fusion convolutions for different levels
        self.fusion_low = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels, 3, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_mid = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_high = nn.Sequential(
            nn.Conv2d(high_channels * 2, high_channels, 3, padding=1),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(low_channels + mid_channels + high_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, query_feats, support_feats, target_size):
        """
        query_feats: dict with 'low', 'mid', 'high' keys
        support_feats: dict with 'low', 'mid', 'high' keys
        target_size: (H, W) for output
        """
        # Process low-level features
        query_low = query_feats['low']
        support_low = support_feats['low']
        support_low_up = F.interpolate(
            support_low, size=query_low.shape[-2:], 
            mode='bilinear', align_corners=True
        )
        fused_low = self.fusion_low(torch.cat([query_low, support_low_up], dim=1))
        
        # Process mid-level features
        query_mid = query_feats['mid']
        support_mid = support_feats['mid']
        support_mid_up = F.interpolate(
            support_mid, size=query_mid.shape[-2:], 
            mode='bilinear', align_corners=True
        )
        fused_mid = self.fusion_mid(torch.cat([query_mid, support_mid_up], dim=1))
        
        # Process high-level features
        query_high = query_feats['high']
        support_high = support_feats['high']
        support_high_up = F.interpolate(
            support_high, size=query_high.shape[-2:], 
            mode='bilinear', align_corners=True
        )
        fused_high = self.fusion_high(torch.cat([query_high, support_high_up], dim=1))
        
        # Coarse-grained information (pooling and upsampling)
        coarse_low = self.pool_coarse(fused_low)
        coarse_mid = self.pool_coarse(fused_mid)
        coarse_high = self.pool_coarse(fused_high)
        
        # Fine-grained information (pooling and upsampling)
        fine_low = self.pool_fine(fused_low)
        fine_mid = self.pool_fine(fused_mid)
        fine_high = self.pool_fine(fused_high)
        
        # Upsample all to target size for concatenation
        size_ref = fused_high.shape[-2:]
        
        fused_low_up = F.interpolate(fused_low, size=size_ref, mode='bilinear', align_corners=True)
        fused_mid_up = F.interpolate(fused_mid, size=size_ref, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        combined = torch.cat([fused_low_up, fused_mid_up, fused_high], dim=1)
        
        # Final fusion
        output = self.final_fusion(combined)
        
        # Upsample to target size
        output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=True)
        
        return output
