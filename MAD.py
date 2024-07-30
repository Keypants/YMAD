import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d



class DASAttentionGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DASAttentionGate, self).__init__()

        # Depthwise Separable Convolution
        self.dsc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),     # BatchNorm2d
            nn.ReLU(inplace=True)
        )

        # Deformable Convolution
        self.offset_conv = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)  # 2 * 3 * 3 = 18 for 2D offset

        self.de_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, out_channels)  # GroupNorm as an alternative to LayerNorm
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.dsc(x)
        shortcut = x
        offset = self.offset_conv(x)
        x = self.de_conv(x, offset)
        x = self.norm(x)
        x = self.sigmoid(x)
        return x * shortcut + shortcut


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

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, channel, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        out = avg_out + max_out
        return self.sigmoid(out) * shortcut


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * shortcut

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )] # 1

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))  # 19
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.last_channel, num_classes)
        self.DAS_1 = DASAttentionGate(32,160)
        self.DAS_2 = DASAttentionGate(480, 160)
        self.ca = ChannelAttention(self.last_channel + 160, 4)
        self.sa = SpatialAttention(kernel_size=7)
        self.ppd = aggregation(160)
        # self.conv_1 = nn.Conv2d(32, 160, 1)
        # self.conv_2 = nn.Conv2d(480, 160, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.project = nn.Conv2d(self.last_channel + 160, self.last_channel, kernel_size=1)

    def forward(self, x):
        # input [B, 3, 256, 256]
        # x = self.features(x)   # [B, 1280, 8, 8]
        out = []
        for i, module in enumerate(self.features):
            x = module(x)
            out.append(x)
        # out[6]      # [32, 32, 32]
        # out[10]     # [64, 16, 16]  # 160 16 16
        # out[13]     # [96, 16, 16]
        # out[16]     # [160, 8, 8]   # 480 8 8
        # out[17]     # [320, 8, 8]
        #
        x_6 = out[6]   # 32 32 32
        x_10_13 = torch.cat([out[10], out[13]], dim=1)  # 160 16 16
        x_16_17 = torch.cat([out[16], out[17]], dim=1)  # 480 8 8

        x_6 = self.DAS_1(x_6)
        x_16_17 = self.DAS_2(x_16_17)
        #
        fuse = self.ppd(x_16_17, x_10_13, x_6) # fuse [8,160,32,32]
        fuse_avg = self.avg_pool(fuse) # [8,160,8,8]
        # 进一步加强，甚至有降噪的作用
        x = self.sa(self.ca(torch.cat([x, fuse_avg], dim=1)))  # modify
        x = self.project(x)
        x = x.mean([2, 3])     # [B, 1280]
        x = self.classifier(x)  # [B, n_classes]
        return x


# 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'    (have a little question)
# 'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1'
def mad(pretrained=False, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        param_state_dict = torch.hub.load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1',
            progress=progress)  # 参数字典
        model_dict = model.state_dict()  # 模型字典
        state_dict = {k: v for k, v in param_state_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        return model

