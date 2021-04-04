'''
CNN building blocks.
Taken from https://github.com/shubhtuls/factored3d/
'''
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import math

## 2D convolution layers
class conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, batch_norm, activation, kernel_size=3, stride=1):
        super(conv2d, self).__init__()

        use_bias = True
        if batch_norm:
            use_bias = False

        modules = []   
        modules.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=use_bias))
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_planes))
        if activation:
            modules.append(activation)

        self.net=nn.Sequential(*modules)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        return self.net(x)

class deconv2d(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(deconv2d, self).__init__()

        #self.net = nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True), 
        #                         nn.ReLU(inplace=True))

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.net = nn.Sequential(conv2d(in_planes=in_planes, out_planes=out_planes, batch_norm=False, activation=False, kernel_size=3, stride=1))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.upsample(x)
        return self.net(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_planes):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_planes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_conv_block(self, in_planes):
        conv_block = []
        conv_block += [conv2d(in_planes=in_planes, out_planes=in_planes, batch_norm=False, activation=nn.ReLU(), kernel_size=3, stride=1)]
        conv_block += [conv2d(in_planes=in_planes, out_planes=in_planes, batch_norm=False, activation=False, kernel_size=3, stride=1)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Cascade_resnet_blocks(nn.Module):
    def __init__(self, in_planes, n_blocks):
        super(Cascade_resnet_blocks, self).__init__()

        resnet_blocks = []
        for i in range(n_blocks):
            resnet_blocks += [ResnetBlock(in_planes)]

        self.net = nn.Sequential(*resnet_blocks)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)
