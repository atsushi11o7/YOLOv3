#Darknet53の実装

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

class ResidualBlock(nn.Module):
    def  __init__(self, num_filters, num_blocks):
        super().__init__()

        self.block_list = nn.ModuleList()
        for _ in range(num_blocks):
            self.block_list.append(
               nn.Sequential(
                nn.Conv2d(num_filters, num_filters//2, 1, 1),
                nn.BatchNorm2d(num_filters//2),
                nn.LeakyReLU(),
                nn.Conv2d(num_filters//2, num_filters, 3, 1, 1),
                nn.BatchNorm2d(num_filters),
                nn.LeakyReLU()
            )
        )

    def forward(self, x):
        for block in self.block_list:
            x = x + block(x)

        return x

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        self.darknet26 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            ResidualBlock(64, 1),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            ResidualBlock(128, 2),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            ResidualBlock(256, 8)
        )

        self.darknet43 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            ResidualBlock(512, 8)
        )

        self.darknet52 = nn.Sequential(

            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            ResidualBlock(1024, 4)
        )

        self.darknet_final_layer = nn.Sequential(
            nn.Conv2d(1024, 1000, 1, 1, 0),
            nn.BatchNorm2d(1000),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.darknet26(x)
        x = self.darknet43(x)
        x = self.darknet52(x)
        x = self.darknet_final_layer(x)
        return x