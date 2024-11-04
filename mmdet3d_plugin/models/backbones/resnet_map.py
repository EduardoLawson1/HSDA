# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from torch import nn
from mmcv.cnn import constant_init, kaiming_init
from torch.nn import functional as F
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import torch.utils.checkpoint as checkpoint

from mmdet.models import BACKBONES
from mmdet3d.models.backbones.resnet import ResNetForBEVDet


@BACKBONES.register_module()
class ResNetForBEVDet_MAP(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
    
    def forward(self, x):
        feats = []
        x_tmp = x
        feats.append(x_tmp)
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
    
@BACKBONES.register_module()
class ResNetForBEVDet_MAP_Attn(ResNetForBEVDet_MAP):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        self.attentions = nn.Sequential(
            nn.AvgPool2d(kernel_size=128),
            nn.Conv2d(numC_input,numC_input//8,1),
            nn.ReLU(numC_input//8),
            nn.Conv2d(numC_input//8,numC_input,1),
            nn.ReLU(numC_input),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        feats = []
        x_attention = self.attentions(x)
        x = x*x_attention
        x_tmp = x
        feats.append(x_tmp)
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Comb(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        self.attentions = nn.Sequential(
            nn.AvgPool2d(kernel_size=128),
            nn.Conv2d(cur_input,cur_input,9,padding=4),
            nn.Sigmoid(),
        )

        self.attention_x = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_xy = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_y = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_s = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = []
        
        x_attention = self.attentions(x)
        x = x*x_attention
        x_conv = self.attention_x(x)
        y = self.attention_y(x)
        y_conv = self.attention_xy(y)
        A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        y_weighted = A*y+y
        x_tmp = torch.cat((x,y_weighted),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_ECA(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(int(numC_input/2),num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        # self.attention_x = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_xy = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_y = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_s = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = []
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # 8, 64, 1, 1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        x_tmp = x * y.expand_as(x)  ### torch.Size([8, 64, 128, 128])

        # x_attention = self.attentions(x)
        # x = x*x_attention
        # x_conv = self.attention_x(x)
        # y = self.attention_y(x)
        # y_conv = self.attention_xy(y)
        # A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        # y_weighted = A*y+y
        # x_tmp = torch.cat((x,y_weighted),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_SASA(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(int(numC_input),num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)

        in_channels = numC_input
        out_channels = in_channels
        kernel_size = 3
        bias = False

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = 1
        self.groups = 1

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        feats = []
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        x_tmp = out

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_TA(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(int(numC_input),num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        
        gate_channels = numC_input
        reduction_ratio=16
        pool_types=["avg", "max"]
        no_spatial=False

        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

        # in_channels = numC_input
        # out_channels = in_channels
        # kernel_size = 3
        # bias = False

        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = 1
        # self.padding = 1
        # self.groups = 1

        # assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        # self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        feats = []

        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        
        x_tmp = x_out

        # batch, channels, height, width = x.size()

        # padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # q_out = self.query_conv(x)
        # k_out = self.key_conv(padded_x)
        # v_out = self.value_conv(padded_x)

        # k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        # k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        # k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        # q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        # out = q_out * k_out
        # out = F.softmax(out, dim=-1)
        # out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # x_tmp = out

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class ResNetForBEVDet_Attn_SE(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(int(numC_input),num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input)
        reduction=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(cur_input, cur_input // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(cur_input // reduction, cur_input, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feats = []

        b, c, _, _ = x.size() # torch.Size([8, 64, 128, 128])    
        y = self.avg_pool(x).view(b, c) # torch.Size([8, 64])
        y = self.fc(y).view(b, c, 1, 1)
        x_tmp = x * y.expand_as(x)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Comb_ECA(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        self.attention_x = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_xy = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_y = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_s = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = []
        
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # 8, 64, 1, 1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y.expand_as(x)

        # x_attention = self.attentions(x)
        # x = x*x_attention
        x_conv = self.attention_x(x)
        y = self.attention_y(x)
        y_conv = self.attention_xy(y)
        A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        y_weighted = A*y+y
        x_tmp = torch.cat((x,y_weighted),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Spatial(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)

        self.attention_x = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_xy = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_y = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_s = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = []
        
        # x_attention = self.attentions(x)
        # x = x*x_attention
        x_conv = self.attention_x(x)
        y = self.attention_y(x)
        y_conv = self.attention_xy(y)
        A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        y_weighted = A*y+y
        x_tmp = torch.cat((x,y_weighted),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Spatial_Hierarchy(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=[256,1024,2048], stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        
        assert len(num_layer)==len(stride)
        # num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
        #     if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'Basic':
            curr_numC = [128,512,1024]
            for i in range(len(num_layer)):
                # print(i,curr_numC,'curr_numC')
                layer=[BasicBlock(curr_numC[i], num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC[i],num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                # curr_numC= num_channels[i]
                layer.extend([BasicBlock(num_channels[i], num_channels[i], norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        cur_input = int(numC_input/2)
        self.attention_x = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_xy = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_y = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.attention_s = nn.Sequential(
            nn.Conv2d(cur_input,cur_input,1),
            nn.ReLU(cur_input),
        )
        self.Sigmoid = nn.Sigmoid()

        cur_input_late = int(numC_input*2)

        self.attention_x_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_xy_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_y_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_s_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        # self.concat_late = nn.Sequential(
        #     nn.Conv2d(cur_input_late*2,cur_input_late,1),
        #     nn.ReLU(cur_input_late),
        # )
        
    def forward(self, x):
        feats = []
        
        x_conv = self.attention_x(x)
        y = self.attention_y(x)
        y_conv = self.attention_xy(y)
        A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        y_weighted = A*y+y
        x_tmp = torch.cat((x,y_weighted),1)

        for lid, layer in enumerate(self.layers):
            # print(lid,x_tmp.shape)
            if lid == 1:
                x_conv = self.attention_x_late(x_tmp)
                y = self.attention_y_late(x_tmp)
                y_conv = self.attention_xy_late(y)
                A = self.Sigmoid(self.attention_s_late(x_conv+y_conv))
                y_weighted = A*y+y
                x_tmp = torch.cat((x_tmp,y_weighted),1)
                # x_tmp = self.concat_late(x_tmp)
                # print(lid,'attention',x_tmp.shape)
                # assert 0
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            
            if lid in self.backbone_output_ids:
                # print(lid,x_tmp.shape)
                feats.append(x_tmp)
        # assert 0
            
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Spatial_nonlocal(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        # self.attentions = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=128),
        #     nn.Conv2d(cur_input,cur_input,9,padding=4),
        #     nn.Sigmoid(),
        # )

        self.attention_query = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.attention_key = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.attention_value = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.Sigmoid = nn.Sigmoid()

        self.attention_mix = nn.Sequential(
            nn.Conv2d(cur_input,numC_input,1),
            nn.ReLU(numC_input),
        )

        # self.attention_x = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_xy = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_y = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_s = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        
    def forward(self, x):
        # x shape torch.Size([8, 64, 128, 128])
        feats = []
        
        # x_conv = self.attention_x(x)
        # y = self.attention_y(x)
        # y_conv = self.attention_xy(y)
        # A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        # y_weighted = A*y+y
        # x_tmp = torch.cat((x,y_weighted),1)
        # torch.Size([8, 128, 128, 128])

        x_query = self.attention_query(x).permute(0,2,3,1)
        B, H, W, C = x_query.shape
        x_query = torch.reshape(x_query,[B, H*W, C])
        x_key = self.attention_key(x)
        x_key = torch.reshape(x_key, [B, C, H*W])
        x_mix = self.Sigmoid(torch.matmul(x_query, x_key)) # torch.Size([B, 16384, 16384])
        
        x_value = self.attention_value(x)
        x_value = torch.reshape(x_value,[B, H*W, C]) # torch.Size([B, 16384, 32])
        
        x_tri_mix = torch.matmul(x_mix, x_value)
        x_tri_mix = torch.reshape(x_tri_mix, [B, C, H, W])

        x_tri_mix = self.attention_mix(x_tri_mix)
        x_add = x_tri_mix+x
     
        x_tmp = x_add

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        
        return feats

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

@BACKBONES.register_module()
class ResNetForBEVDet_Attn_GCNet(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input)

        self.inplanes = cur_input
        self.planes = cur_input
        self.pooling_type = 'att'
        self.fusion_types = ('channel_add', )
        if self.pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in self.fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in self.fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)
    
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context
        
    def forward(self, x):
        # x shape torch.Size([8, 64, 128, 128])
        feats = []

        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        
        x_tmp = out

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        
        return feats


@BACKBONES.register_module()
class ResNetForBEVDet_Attn_Channel(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(int(numC_input/2),num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        self.attentions = nn.Sequential(
            nn.AvgPool2d(kernel_size=128),
            nn.Conv2d(cur_input,cur_input,9,padding=4),
            nn.Sigmoid(),
        )

        # self.attention_x = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_xy = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_y = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.attention_s = nn.Sequential(
        #     nn.Conv2d(cur_input,cur_input,1),
        #     nn.ReLU(cur_input),
        # )
        # self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feats = []
        
        x_attention = self.attentions(x)
        x = x*x_attention
        # x_conv = self.attention_x(x)
        # y = self.attention_y(x)
        # y_conv = self.attention_xy(y)
        # A = self.Sigmoid(self.attention_s(x_conv+y_conv))
        # y_weighted = A*y+y
        # x_tmp = torch.cat((x,y_weighted),1)
        x_tmp = x

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
    
class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 nn.BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out

class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = nn.BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = nn.BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

@BACKBONES.register_module()
class ResNetForBEVDet_graph(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)

        self.dualgcn = DualGCN(numC_input)

    def forward(self, x):
        feats = []

        x = self.dualgcn(x)
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_graph_simple(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/8)

        self.attention_query = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.attention_key = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.attention_value = nn.Sequential(
            nn.Conv2d(numC_input,cur_input,1),
            nn.ReLU(cur_input),
        )

        self.Sigmoid = nn.Sigmoid()

        self.attention_mix = nn.Sequential(
            nn.Conv2d(cur_input,numC_input,1),
            nn.ReLU(numC_input),
        )
        
    def forward(self, x):
        # x shape torch.Size([8, 64, 128, 128])
        feats = []

        x_query = self.attention_query(x).permute(0,2,3,1)
        B, H, W, C = x_query.shape
        x_query = torch.reshape(x_query,[B, H*W, C])
        x_key = self.attention_key(x)
        x_key = torch.reshape(x_key, [B, C, H*W])
        x_mix = self.Sigmoid(torch.matmul(x_query, x_key)) # torch.Size([B, 16384, 16384])
        
        x_value = self.attention_value(x)
        x_value = torch.reshape(x_value,[B, H*W, C]) # torch.Size([B, 16384, 32])
        
        x_tri_mix = torch.matmul(x_mix, x_value)
        x_tri_mix = torch.reshape(x_tri_mix, [B, C, H, W])

        x_tri_mix = self.attention_mix(x_tri_mix)
        x_add = x_tri_mix+x
     
        x_tmp = x_add

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        
        return feats

import torch
import torch.nn as nn

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

@BACKBONES.register_module()
class ResNetForBEVDet_graph_feature(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)

        self.GloRe_Unit_2D = GloRe_Unit_2D(numC_input, cur_input)
    
    def forward(self, x):
        feats = []
        x = self.GloRe_Unit_2D(x)
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

class GCN_node(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN_node, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_node, num_node, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h.permute(0, 2, 1).contiguous())).permute(0, 2, 1)
        return h

class GCN_channel(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN_channel, self).__init__()
        self.conv1 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x)
        return h


class GloRe_Unit_concat(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit_concat, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN_node(num_state=self.num_s, num_node=self.num_n)
        self.gcn_channel = GCN_channel(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)
        x_n_rel_channel = self.gcn_channel(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel+x_n_rel_channel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out

class GloRe_Unit_2D_concat(GloRe_Unit_concat):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D_concat, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

@BACKBONES.register_module()
class ResNetForBEVDet_graph_concat(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)

        self.GloRe_Unit_2D_concat = GloRe_Unit_2D_concat(numC_input, cur_input)
    
    def forward(self, x):
        feats = []
        x = self.GloRe_Unit_2D_concat(x)
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats



@BACKBONES.register_module()
class ResNetForBEVDet_graph_spatial(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        node = 8

        self.shrink_origin = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)

        self.shrink_graph = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(cur_input, cur_input, kernel_size=1)

        self.extend = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        # x shape torch.Size([8, 64, 128, 128])
        # [batch_size,channel,height,weight]
        feats = []

        n, channel, h, w = x.shape
        shrink_origin = self.shrink_origin(x)
        h_shrink, w_shrink = shrink_origin.shape[2:]
        
        shrink_graph = self.shrink_graph(x).view(n,channel,-1)
        shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

        shrink = shrink_origin + shrink_graph

        shrink_extend = self.extend(shrink)

        x_tmp = torch.cat((x,shrink_extend),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_graph_spatial_hierachy(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=[256,1024,2048], stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        
        assert len(num_layer)==len(stride)
        # num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
        #     if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'Basic':
            curr_numC = [128,512,1024]
            for i in range(len(num_layer)):
                # print(i,curr_numC,'curr_numC')
                layer=[BasicBlock(curr_numC[i], num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC[i],num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                # curr_numC= num_channels[i]
                layer.extend([BasicBlock(num_channels[i], num_channels[i], norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)
        
        
        cur_input = int(numC_input/2)
        node = 8
        self.shrink_origin = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)
        self.shrink_graph = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(cur_input, cur_input, kernel_size=1)
        self.extend = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        cur_input_late = int(numC_input*2)
        self.shrink_origin_late = nn.Conv2d(cur_input_late, cur_input_late, kernel_size=1,stride=8)
        self.shrink_graph_late = nn.Conv2d(cur_input_late, cur_input_late, kernel_size=1,stride=8)
        self.graph_1_late = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2_late = nn.Conv1d(cur_input_late, cur_input_late, kernel_size=1)
        self.extend_late = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        # x shape torch.Size([8, 64, 128, 128])
        # [batch_size,channel,height,weight]
        feats = []

        n, channel, h, w = x.shape
        shrink_origin = self.shrink_origin(x)
        h_shrink, w_shrink = shrink_origin.shape[2:]
        
        shrink_graph = self.shrink_graph(x).view(n,channel,-1)
        shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

        shrink = shrink_origin + shrink_graph

        shrink_extend = self.extend(shrink)

        x_tmp = torch.cat((x,shrink_extend),1)

        for lid, layer in enumerate(self.layers):
            if lid == 1:
                n, channel, h, w = x_tmp.shape
                shrink_origin_late = self.shrink_origin_late(x_tmp)
                h_shrink, w_shrink = shrink_origin_late.shape[2:]
                
                shrink_graph_late = self.shrink_graph_late(x_tmp).view(n,channel,-1)
                shrink_graph_late = self.graph_1_late(shrink_graph_late.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
                shrink_graph_late = self.graph_2_late(self.relu(shrink_graph_late)).view(n,channel,h_shrink,-1)

                shrink_late = shrink_origin_late + shrink_graph_late

                shrink_extend_late = self.extend_late(shrink_late)

                x_tmp = torch.cat((x_tmp,shrink_extend_late),1)
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_graph_spatial_4node(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        cur_input = int(numC_input/2)
        node = 4

        self.shrink_origin = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=32)

        self.shrink_graph = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=32)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(cur_input, cur_input, kernel_size=1)

        self.extend = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        feats = []

        n, channel, h, w = x.shape
        shrink_origin = self.shrink_origin(x)
        h_shrink, w_shrink = shrink_origin.shape[2:]
        
        shrink_graph = self.shrink_graph(x).view(n,channel,-1)
        shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

        shrink = shrink_origin + shrink_graph

        shrink_extend = self.extend(shrink)

        x_tmp = torch.cat((x,shrink_extend),1)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@BACKBONES.register_module()
class ResNetForBEVDet_graph_spatial_SA_hierarchy(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=[256,1024,2048], stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        
        assert len(num_layer)==len(stride)
        # num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
        #     if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'Basic':
            curr_numC = [128,512,1024]
            for i in range(len(num_layer)):
                # print(i,curr_numC,'curr_numC')
                layer=[BasicBlock(curr_numC[i], num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC[i],num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                # curr_numC= num_channels[i]
                layer.extend([BasicBlock(num_channels[i], num_channels[i], norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        cur_input = int(numC_input/2)
        node = 8

        self.shrink_origin = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)

        self.shrink_graph = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(cur_input, cur_input, kernel_size=1)

        self.extend = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
        self.Sigmoid = nn.Sigmoid()

        cur_input_late = int(numC_input*2)

        self.attention_x_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_xy_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_y_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_s_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )

    def forward(self, x):
        feats = []

        n, channel, h, w = x.shape
        shrink_origin = self.shrink_origin(x)
        h_shrink, w_shrink = shrink_origin.shape[2:]
        
        shrink_graph = self.shrink_graph(x).view(n,channel,-1)
        shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

        shrink = shrink_origin + shrink_graph

        shrink_extend = self.extend(shrink)

        x_tmp = torch.cat((x,shrink_extend),1)

        for lid, layer in enumerate(self.layers):
            # print(lid,x_tmp.shape)
            if lid == 1:
                x_conv = self.attention_x_late(x_tmp)
                y = self.attention_y_late(x_tmp)
                y_conv = self.attention_xy_late(y)
                A = self.Sigmoid(self.attention_s_late(x_conv+y_conv))
                y_weighted = A*y+y
                x_tmp = torch.cat((x_tmp,y_weighted),1) # 8, 512, 64, 64
                # x_tmp = self.concat_late(x_tmp)
                # print(lid,'attention',x_tmp.shape)
                # assert 0
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            
            if lid in self.backbone_output_ids:
                # print(lid,x_tmp.shape)
                feats.append(x_tmp)

        # for lid, layer in enumerate(self.layers):
        #     if self.with_cp:
        #         x_tmp = checkpoint.checkpoint(layer, x_tmp)
        #     else:
        #         x_tmp = layer(x_tmp)
        #     if lid in self.backbone_output_ids:
        #         feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class ResNetForBEVDet_SA_graph_spatial_hierarchy(ResNetForBEVDet):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=[256,1024,2048], stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',):
        super().__init__(numC_input,num_layer=num_layer,num_channels=num_channels,stride=stride,backbone_output_ids=backbone_output_ids,norm_cfg=norm_cfg,with_cp=with_cp,block_type=block_type)
        
        assert len(num_layer)==len(stride)
        # num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
        #     if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'Basic':
            curr_numC = [128,512,1024]
            for i in range(len(num_layer)):
                # print(i,curr_numC,'curr_numC')
                layer=[BasicBlock(curr_numC[i], num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC[i],num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                # curr_numC= num_channels[i]
                layer.extend([BasicBlock(num_channels[i], num_channels[i], norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        cur_input = int(numC_input*2)
        node = 4

        self.shrink_origin = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)

        self.shrink_graph = nn.Conv2d(cur_input, cur_input, kernel_size=1,stride=16)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(cur_input, cur_input, kernel_size=1)

        self.extend = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
        self.Sigmoid = nn.Sigmoid()

        cur_input_late = int(numC_input/2)

        self.attention_x_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_xy_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_y_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )
        self.attention_s_late = nn.Sequential(
            nn.Conv2d(cur_input_late,cur_input_late,1),
            nn.ReLU(cur_input_late),
        )

    def forward(self, x):
        feats = [] # x shape torch.Size([6, 64, 128, 128]) 
        x_conv = self.attention_x_late(x)
        #print(x_conv.shape)
        #assert 0
        y = self.attention_y_late(x)
        y_conv = self.attention_xy_late(y)
        A = self.Sigmoid(self.attention_s_late(x_conv+y_conv))
        y_weighted = A*y+y
        x_tmp = torch.cat((x,y_weighted),1) # 6, 128, 128, 128
        # x_tmp = self.concat_late(x_tmp)
        # print(lid,'attention',x_tmp.shape)
        # assert 0

        


        for lid, layer in enumerate(self.layers):
            # print(lid,x_tmp.shape)
            if lid == 1:
                n, channel, h, w = x_tmp.shape # torch.Size([6, 256, 64, 64]) 
                shrink_origin = self.shrink_origin(x_tmp)
                h_shrink, w_shrink = shrink_origin.shape[2:]
                
                shrink_graph = self.shrink_graph(x_tmp).view(n,channel,-1)
                shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
                shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

                shrink = shrink_origin + shrink_graph

                shrink_extend = self.extend(shrink)

                x_tmp = torch.cat((x_tmp,shrink_extend),1) # torch.Size([6, 512, 64, 64])

            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            
            if lid in self.backbone_output_ids:
                # print(lid,x_tmp.shape)
                feats.append(x_tmp)

        # for lid, layer in enumerate(self.layers):
        #     if self.with_cp:
        #         x_tmp = checkpoint.checkpoint(layer, x_tmp)
        #     else:
        #         x_tmp = layer(x_tmp)
        #     if lid in self.backbone_output_ids:
        #         feats.append(x_tmp)
        return feats
