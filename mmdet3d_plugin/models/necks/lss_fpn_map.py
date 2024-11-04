# Copyright (c) Phigent Robotics. All rights reserved.

import imp
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.models import NECKS
from mmdet3d.models.necks import FPN_LSS


@NECKS.register_module()
class FPN_LSS_MAP(FPN_LSS):
    def __init__(self, in_channels, out_channels, 
                 # in_channels_detect=640, 
                 scale_factor=4,
                 input_feature_index = (0,1,2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None, add_origin=None, 
                #  separate=None
                 ):
        super().__init__(in_channels, out_channels, scale_factor=scale_factor,
                 input_feature_index = input_feature_index,
                 norm_cfg=norm_cfg,
                 extra_upsample=extra_upsample,
                 lateral=lateral)
        self.up_mid = nn.Upsample(scale_factor=scale_factor//2, mode='bilinear', align_corners=True)
        self.add_origin = add_origin
        # self.separate = separate
        # channels_factor = 2 if self.extra_upsample else 1
        # self.conv_detect = nn.Sequential(
        #     nn.Conv2d(in_channels_detect, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
        #               kernel_size=3, padding=1, bias=False),
        #     build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
        #     nn.ReLU(inplace=True),
        # )
    
    def forward(self, feats):
        x2, x1, x0 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]], feats[self.input_feature_index[2]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x0 = self.up(x0)
        x1 = self.up_mid(x1)
        x_0 = torch.cat([x2, x1, x0], dim=1)
        x = self.conv(x_0)
        # x_cat = torch.cat([x2, x1, x0], dim=1)
        # x = self.conv(x_cat)
        if self.extra_upsample:
            x = self.up2(x)
        if self.add_origin:
            x = torch.cat([feats[0],x], dim=1)
        # if self.separate:
        #     x_detect = self.conv_detect(torch.cat([x2, x0], dim=1))
        #     x_detect = self.up2(x_detect)
        #     x = torch.cat([x, x_detect], dim=1)
            
        return x