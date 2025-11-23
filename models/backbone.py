"""
Backbone networks for FSS Network
ResNet50 with PSPNet feature extractor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PSPModule(nn.Module):
    """Pyramid Pooling Module for PSPNet"""
    
    def __init__(self, in_channels, out_channels=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, out_channels, size) for size in sizes
        ])
        self.bottleneck = nn.Conv2d(
            in_channels + len(sizes) * out_channels, 
            out_channels, 
            kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        
    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.interpolate(
            stage(x), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=True
        ) for stage in self.stages]
        
        out = torch.cat([x] + priors, dim=1)
        out = self.bottleneck(out)
        out = self.relu(out)
        return out


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone with PSPNet feature extractor"""
    
    def __init__(self, pretrained=True, freeze=True):
        super(ResNet50Backbone, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # PSPNet module
        self.psp = PSPModule(2048, 512)
        
        # Freeze backbone if specified
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            # Keep PSP module trainable
            for param in self.psp.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x1 = self.layer1(x)   # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        
        # PSPNet feature extraction
        features = self.psp(x4)
        
        return {
            'layer1': x1,
            'layer2': x2,
            'layer3': x3,
            'layer4': x4,
            'psp': features
        }
