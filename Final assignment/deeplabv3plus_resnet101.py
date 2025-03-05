"""
DeepLabV3+ Model (ResNet-101 backbone)
Adapted from https://github.com/VainF/DeepLabV3Plus-Pytorch
License: MIT (See full text at the bottom)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# --------------------------------------------------------------------------------
# Part 1: ResNet backbone (resnet.py) 
# --------------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 output_stride=16,
                 norm_layer=None,
                 pretrained=False,  # We won't load pretrained weights here
                 ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        if output_stride == 16:
            strides = (1, 2, 2, 1)
            dilations = (1, 1, 1, 2)
        elif output_stride == 8:
            strides = (1, 2, 1, 1)
            dilations = (1, 1, 2, 4)
        else:
            raise NotImplementedError

        # First layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], 
                               padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)

        # Build layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1,
                                       dilation=dilations[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2],
                                       dilation=dilations[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3],
                                       dilation=dilations[2], norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=dilations[3], norm_layer=norm_layer)

        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)   # (B, 64, H/1, W/1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (B, 64, H/2, W/2)

        x = self.layer1(x)  # (B, 256, H/2, W/2)
        x = self.layer2(x)  # (B, 512, H/4, W/4)
        x = self.layer3(x)  # (B, 1024, H/8, W/8)
        x = self.layer4(x)  # (B, 2048, H/16, W/16)

        return x

def build_resnet101(output_stride=16, norm_layer=None):
    """Construct a ResNet-101 backbone."""
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   output_stride=output_stride,
                   norm_layer=norm_layer,
                   pretrained=False)
    return model


# --------------------------------------------------------------------------------
# Part 2: ASPP (Atrous Spatial Pyramid Pooling) and Decoder (DeepLabV3+)
# --------------------------------------------------------------------------------

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling submodule."""
    def __init__(self, in_channels, out_channels, dilation, norm_layer):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, norm_layer=None):
        super(ASPP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = ASPPModule(in_channels, out_channels, dilations[0], norm_layer)
        self.conv3 = ASPPModule(in_channels, out_channels, dilations[1], norm_layer)
        self.conv4 = ASPPModule(in_channels, out_channels, dilations[2], norm_layer)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_pool(x)
        # Upsample x5 to match x size
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x_out = self.conv_out(x_cat)
        return x_out


# --------------------------------------------------------------------------------
# Part 3: DeepLabV3+ model
# --------------------------------------------------------------------------------

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, norm_layer=None):
        super(DeepLabHeadV3Plus, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.reduce_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            norm_layer(48),
            nn.ReLU(inplace=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.reduce_low_level(low_level_feat)
        # Upsample x to match low-level feature spatial size
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.last_conv(x)
        return x

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with a ResNet-101 backbone.
    num_classes: number of output channels in your segmentation (e.g. 19 for Cityscapes)
    output_stride: 16 (standard) or 8
    """
    def __init__(self, num_classes=19, output_stride=16, norm_layer=None):
        super(DeepLabV3Plus, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 1. Backbone (ResNet-101)
        self.backbone = build_resnet101(output_stride, norm_layer=norm_layer)

        # 2. ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256,
                         dilations=(12, 24, 36), norm_layer=norm_layer)

        # 3. Decoder
        #   low_level_channels = 256 from layer1 (ResNet).
        self.decoder = DeepLabHeadV3Plus(in_channels=256,
                                         low_level_channels=256,
                                         num_classes=num_classes,
                                         norm_layer=norm_layer)

    def forward(self, x):
        # ResNet forward
        input_size = x.shape[2:]
        # layer0
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        low_level_feat = x  # for DeepLabV3+ we often use this feature from the very beginning
        x = self.backbone.maxpool(x)

        # layer1 -> low-level feature for the decoder
        x = self.backbone.layer1(x)
        low_level_feat = x  # we update low_level_feat here (often from layer1)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


# --------------------------------------------------------------------------------
# Part 4: Quick test
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    model = DeepLabV3Plus(num_classes=19, output_stride=16)
    model.eval()
    inp = torch.randn(1, 3, 512, 512)  # simulate a batch of size 1
    out = model(inp)
    print("Output shape:", out.shape)  # should be (1, 19, 512, 512)


# --------------------------------------------------------------------------------
# MIT License from original repository (https://github.com/VainF/DeepLabV3Plus-Pytorch)
# --------------------------------------------------------------------------------
"""
MIT License

Copyright (c) 2019 VainF

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

... [full MIT text as in original] ...
"""
