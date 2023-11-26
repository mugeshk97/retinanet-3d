import math
import os
import sys

import torch
import torch.nn as nn

from config import data_config


class PredictionHead(nn.Module):
    def __init__(self, input_channel, head_conv):
        super(PredictionHead, self).__init__()

        self.input_channel = input_channel
        self.head_conv = head_conv

        self.hm_cen = self._create_head(data_config.num_classes)
        self.cen_offset = self._create_head(data_config.num_center_offset)
        self.z_coor = self._create_head(data_config.num_z)
        self.dim = self._create_head(data_config.num_dim)
        self.direction = self._create_head(data_config.num_direction)

    def _create_head(self, num_output_channels):
        if self.head_conv > 0:
            head = nn.Sequential(
                nn.Conv2d(self.input_channel, self.head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, num_output_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            head = nn.Conv2d(self.input_channel, num_output_channels, kernel_size=1, stride=1, padding=0)

        return head

    def forward(self, x):
        outputs = {
            'hm_cen': self.hm_cen(x),
            'cen_offset': self.cen_offset(x),
            'z_coor': self.z_coor(x),
            'dim': self.dim(x),
            'direction': self.direction(x)
        }
        return outputs
