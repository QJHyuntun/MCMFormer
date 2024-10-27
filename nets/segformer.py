# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DepthSeparableConv(nn.Module):
    """
    深度可分离卷积模块
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MDSPP(nn.Module):
    """
    混合深度可分离金字塔池化模块
    """

    def __init__(self, in_channels, out_channels, sizes):
        super(MDSPP, self).__init__()
        # 定义池化阶段
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                DepthSeparableConv(in_channels, out_channels, kernel_size=1, padding=0)
            ) for size in sizes
        ])
        # 定义瓶颈层
        self.bottleneck = nn.Sequential(
            DepthSeparableConv(len(sizes) * out_channels + in_channels, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        size = x.size()[2:]  # 获取输入的尺寸 (H, W)
        features = [x]
        for stage in self.stages:
            pooled = stage(x)
            # 将池化后的特征图上采样到输入尺寸
            pooled = F.interpolate(pooled, size=size, mode='bilinear', align_corners=False)
            features.append(pooled)
        out = torch.cat(features, dim=1)
        out = self.bottleneck(out)
        return out

    """
    		门控融合机制，自适应权重
    """

class SpatialAttention(nn.Module):
    """
    空间注意力机制模块
    """
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    """
    通道注意力机制模块
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.global_avg_pool(x))
        max_out = self.fc(self.global_max_pool(x))
        return self.sigmoid(avg_out + max_out)

class GatedFusion(nn.Module):
    """
    门控融合模块
    """
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, channel_att, spatial_att):
        gate_weight = self.gate_fc(channel_att)
        fused_att = gate_weight * channel_att + (1 - gate_weight) * spatial_att
        return fused_att

class DynamicJointAttention(nn.Module):
    """
    动态调整的空间-通道联合注意力机制模块，门控融合方式
    """
    def __init__(self, in_channels, kernel_size=7, reduction=16):
        super(DynamicJointAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.gated_fusion = GatedFusion(in_channels)

    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        fused_att = self.gated_fusion(channel_att, spatial_att)
        x = x * fused_att
        return x

class PolarizationAttention(nn.Module):
    """
    极化注意力机制模块
    """
    def __init__(self, in_channels):
        super(PolarizationAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        out = self.fc(out)
        return out



class DynamicJointAttentionC(nn.Module):
    """
    动态调整的空间-通道联合注意力机制模块
    """
    def __init__(self, in_channels, kernel_size=7, reduction=16):
        super(DynamicJointAttentionC, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        x = x * (self.gamma * channel_att + (1 - self.gamma) * spatial_att)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.joint = DynamicJointAttention(embedding_dim)  # 联合注意力机制
        self.jointC = DynamicJointAttentionC(embedding_dim)  # 动态参数
        self.psa = PolarizationAttention(embedding_dim)  # 极化

        # 金字塔池化模块
        self.mpsp = MDSPP(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim // 4,
            sizes=[1, 2, 4, 8]
        )

        # 分类预测层
        self.extra_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)  # 额外卷积层
        self.linear_pred = nn.Conv2d(embedding_dim // 4, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        # 特征投影
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, h, w)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c3 = self.jointC(_c3)
        _c2 = self.jointC(_c2)
        # _c1 = self.psa(_c1)


        # 特征精炼
        _c4 = self.extra_conv(_c4)

        # 将所有特征图进行金字塔池化
        _c = self.mpsp(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
