import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log
from model.Res2Net import res2net50_v1b_26w_4s
from model.base import BaseNetwork
from model.transformer_block import FeedForward2D
from einops import rearrange


class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=64, w=33, fp32fft=True):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class Fuse(nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()
        self.predict3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, xr, dualattention):
        crop_3 = F.interpolate(dualattention, xr.size()[2:], mode='bilinear', align_corners=False)
        re3_feat = self.predict3(torch.cat([xr, crop_3], dim=1))

        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 64, -1, -1).mul(xr)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra3_feat = self.ra2_conv4(x)

        x = ra3_feat + crop_3 + re3_feat

        return x


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True))
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1))
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1))
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1))

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def run_sobel(conv_x, conv_y, conv_z1, conv_z2, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g_z1 = conv_z1(input)
    g_z2 = conv_z2(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + torch.pow(g_z1, 2) + torch.pow(g_z2, 2))
    return torch.sigmoid(g) * input


def get_sobel(in_chan, out_chan):
    #  The code will be open source soon

    return sobel_x, sobel_y, sobel_z1, sobel_z2


class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x + res


class _PositionAttentionModule(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class DFR_later(nn.Module):
    def __init__(self):
        super(DFR_later, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class PAA(nn.Module):
    def __init__(self, in_channel=256, depth=256):
        super(PAA, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block2(x)
        atrous_block12 = self.atrous_block3(x)
        atrous_block18 = self.atrous_block4(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))
        net = net + x

        return net


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=2, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        #  The code will be open source soon
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),)
        self.conv1 = nn.Conv2d(
            d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.SMAT = Attention(256,8)

        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x32 = self.conv1(x)
        x16 = self.conv1(x32)
        x8 = self.conv1(x16)
        x4 = self.conv1(x8)

        x32 = self.SMAT(x32)
        x16 = self.SMAT(x16)
        x8 = self.SMAT(x8)
        x4 = self.SMAT(x4)

        x32 = self.upsample2(x32)
        x16 = self.upsample4(x16)
        x8 = self.upsample8(x8)
        x4 = self.upsample16(x4)

        output = x4 + x8 + x16 + x32
        self_attention = self.output_linear(output)

        return self_attention


class TransformerBlock(nn.Module):
    def __init__(self, patchsize, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel)

    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self_attention
        output = output + self.feed_forward(output)

        return output


class MST(BaseNetwork):
    def __init__(self, in_channel, in_size):
        super(MST, self).__init__()
        self.in_size = in_size

        patchsize = [
            (32, 32),
            (16, 16),
            (8, 8),
            (4, 4),]

        self.t = TransformerBlock(patchsize, in_channel=in_channel)

    def forward(self, x):
        output = self.t(x)

        return output

class ADSTNet(nn.Module):
    def __init__(self, seg_classes):
        super(ADSTNet, self).__init__()
        self.num_class = seg_classes
        self.fft = GlobalFilter(dim=3, h=256, w=129, fp32fft=True)
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.MST = MST(in_channel=256, in_size=64)
        self.PAA = PAA(in_channel=256, depth=256)
        self.DFR_later = DFR_later()

        self.sobel_x1, self.sobel_y1, self.sobel_z11, self.sobel_z12 = get_sobel(256, 1)
        self.sobel_x4, self.sobel_y4, self.sobel_z41, self.sobel_z42 = get_sobel(2048, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample_3 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.erb_db_1 = ERB(256, self.num_class)
        self.erb_db_2 = ERB(512, self.num_class)
        self.erb_db_3 = ERB(1024, self.num_class)
        self.erb_db_4 = ERB(2048, self.num_class)

        self.head = _DAHead(2048 + 256, 2048, aux=False)

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 64)
        self.reduce3 = Conv1x1(1024, 64)
        self.reduce4 = Conv1x1(2048, 64)
        self.reduce5 = Conv1x1(2048, 1)

        self.fuse1 = Fuse()
        self.fuse2 = Fuse()
        self.fuse3 = Fuse()
        self.fuse4 = Fuse()

        self.salhead1 = nn.Conv2d(64, self.num_class, 1)
        self.salhead2 = nn.Conv2d(64, self.num_class, 1)
        self.salhead3 = nn.Conv2d(64, self.num_class, 1)
        self.salhead4 = nn.Conv2d(64, self.num_class, 1)
        self.salhead5 = nn.Conv2d(256, self.num_class, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fft_fea = self.fft(x)
        x0, x1, x2, x3, x4 = self.resnet(x)
        x1_PAA = self.PAA(x1)
        trans = self.MST(x1)
        x1_paa_trans = x1_PAA + trans

        o5 = self.salhead5(x1_paa_trans)
        o5 = F.interpolate(o5, scale_factor=4, mode='bilinear', align_corners=False)

        s1 = run_sobel(self.sobel_x1, self.sobel_y1, self.sobel_z11, self.sobel_z12, x1)
        s4 = run_sobel(self.sobel_x4, self.sobel_y4, self.sobel_z41, self.sobel_z42, x4)
        edge = self.DFR_later(s4, s1)
        edge_att = torch.sigmoid(edge)

# AFCD  --> start
        trans = F.interpolate(x1_paa_trans, x4.size()[2:], mode='bilinear', align_corners=False)
        dual_attention = self.head(torch.cat([trans, x4], dim=1))[0]
        edge_att0 = F.interpolate(edge_att, x0.size()[2:], mode='bilinear', align_corners=False)
        x0a = x0 * edge_att0
        x1a = x1 * edge_att
        edge_att2 = F.interpolate(edge_att, x2.size()[2:], mode='bilinear', align_corners=False)
        x2a = x2 * edge_att2
        edge_att3 = F.interpolate(edge_att, x3.size()[2:], mode='bilinear', align_corners=False)
        x3a = x3 * edge_att3

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)

        dual_attention = self.reduce4(dual_attention)

        c3 = self.fuse3(x3r, dual_attention)
        c2 = self.fuse2(x2r, c3)
        c1 = self.fuse1(x1r, c2)
        c0 = self.fuse1(x0a, c1)
# AFCD  --> end

        o3 = self.salhead3(c3)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.salhead2(c2)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.salhead1(c1)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        o0 = self.salhead1(c0)
        o0 = F.interpolate(o0, scale_factor=2, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o0, o1, o2, o3, o5, self.sigmoid(o0), self.sigmoid(o1), self.sigmoid(o2), self.sigmoid(o3), self.sigmoid(o5), oe