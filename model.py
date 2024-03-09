# YOLOv3の実装

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

from Darknet53 import Darknet53

class ConvBlock(nn.Module):
    def __init__(self, input_block, output_block, num_blocks):
        super(ConvBlock, self).__init__()

        conv_block = nn.ModuleList()
        for block in range(num_blocks):
            in_channels = input_block if block == 0 else output_block
            conv_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, output_block//2, 1, 1),
                    nn.BatchNorm2d(output_block//2),
                    nn.LeakyReLU(),
                    nn.Conv2d(output_block//2, output_block, 3, 1, 1),
                    nn.BatchNorm2d(output_block),
                    nn.LeakyReLU()
                )
            )
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)

class YOLOv3(nn.Module):
    def __init__(self, num_class = 1):
        super(YOLOv3, self).__init__()

        self.darknet26 = Darknet53().darknet26
        self.darknet43 = Darknet53().darknet43
        self.darknet52 = Darknet53().darknet52

        self.conv_block = ConvBlock(1024, 1024, 3)
        self.scale3_YOLO_layer = nn.Conv2d(1024, (3 * (4 + 1 + 3)), 1, 1)

        self.scale2_upsampling = nn.Conv2d(1024, 256, 1, 1)
        self.scale2_conv_block = ConvBlock(768, 512, 3)
        self.scale2_YOLO_layer = nn.Conv2d(512, (3 * (4 + 1 + 3)), 1, 1)

        self.scale1_upsampling = nn.Conv2d(512, 128, 1, 1)
        self.scale1_conv_block = ConvBlock(384, 256, 3)
        self.scale1_YOLO_layer = nn.Conv2d(256, (3 * (4 + 1 + 3)), 1, 1)

        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        x1 = self.darknet26(x)
        x2 = self.darknet43(x1)
        x3 = self.darknet52(x2)

        x3 = self.conv_block(x3)
        scale3_output = self.scale3_YOLO_layer(x3)

        scale2_upsample = self.upsample(self.scale2_upsampling(x3))
        x2 = torch.cat((x2, scale2_upsample), dim=1)
        x2 = self.scale2_conv_block(x2)
        scale2_output = self.scale2_YOLO_layer(x2)

        scale1_upsample = self.upsample(self.scale1_upsampling(x2))
        x1 = torch.cat((x1, scale1_upsample), dim=1)
        x1 = self.scale1_conv_block(x1)
        scale1_output = self.scale1_YOLO_layer(x1)

        return scale3_output, scale2_output, scale1_output