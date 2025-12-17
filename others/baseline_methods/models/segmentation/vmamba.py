import torch
from torch import nn
import torch.nn.functional as F

from others.baseline_methods.models.classification.vmamba import Backbone_VSSM


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dim = dimension

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return torch.cat(x, dim=self.dim)


def autopad(kernel, padding):
    if padding is None:
        return kernel // 2 if isinstance(kernel, int) else [p // 2 for p in kernel]
    else:
        return padding


class Upsample(nn.Module):

    def __init__(self, factor=2) -> None:
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode="bilinear")


class ConvInAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1,
                 bias=False, act=True):
        super(ConvInAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.instancenorm = nn.InstanceNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if x.shape[-1] != 1:
            x = self.instancenorm(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels // 4
        self.cba1 = ConvInAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvInAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvInAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvInAct(in_channels, inter_channels, 1, 1, 0)
        self.out = ConvInAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvInAct(inplanes, planes, 1, 1, 0, act=True)
        self.conv2 = ConvInAct(planes, planes, 3, stride, 1, act=True)
        self.conv3 = ConvInAct(planes, planes * self.expansion, 1, 1, 0, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)  # conv with 1x1 kernel and stride 2
        out = out + residual
        return self.act(out)


class UPerDecoder(nn.Module):
    def __init__(self, encoder_channels, fpn_dim=256, num_classes=21):
        super().__init__()

        # 最终输出 head
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        self.out = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)

    def forward(self, features):
        # 输入特征：list of [P2, P3, P4, P5]，大小分别为128, 64, 32, 16
        sizes = features[0].shape[2:]  # 最终统一上采样到128x128
        fpn_outputs = []

        # 逐层处理并上采样
        for i, (x, block) in enumerate(zip(features[::-1], self.fpn_blocks[::-1])):
            out = block(x, upsample_to=sizes)
            fpn_outputs.append(out)

        # 拼接所有上采样后的特征图
        x = torch.cat(fpn_outputs, dim=1)  # shape: (B, fpn_dim * 4, 128, 128)
        x = self.head(x)  # shape: (B, num_classes, 128, 128)

        # 最终上采样回原始图像尺寸（假设输入为 512x512）
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.out(x)
        return x


class FeaturePyramidNet(nn.Module):

    def __init__(self, input_nc, fpn_dim=256):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in_layer1 = ConvInAct(input_nc, self.fpn_dim, 1, 1, 0)
        self.fpn_in_layer2 = ConvInAct(input_nc * 2, self.fpn_dim, 1, 1, 0)
        self.fpn_in_layer3 = ConvInAct(input_nc * 4, self.fpn_dim, 1, 1, 0)
        self.fpn_out_layer1 = ConvInAct(self.fpn_dim, self.fpn_dim, 3, 1, 1)
        self.fpn_out_layer2 = ConvInAct(self.fpn_dim, self.fpn_dim, 3, 1, 1)
        self.fpn_out_layer3 = ConvInAct(self.fpn_dim, self.fpn_dim, 3, 1, 1)

    def forward(self, pyramid_features):
        """

        """
        f = pyramid_features[3]
        fpn_out_4 = f
        x = self.fpn_in_layer3(pyramid_features[2])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out_3 = self.fpn_out_layer3(f)

        x = self.fpn_in_layer2(pyramid_features[1])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out_2 = self.fpn_out_layer2(f)

        x = self.fpn_in_layer1(pyramid_features[0])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out_1 = self.fpn_out_layer1(f)

        return fpn_out_1, fpn_out_2, fpn_out_3, fpn_out_4


class VMamba_Seg(nn.Module):
    def __init__(self, input_nc, output_nc, channel_first=False, dims=128):
        super().__init__()

        self.vmamba = Backbone_VSSM(in_chans=input_nc, depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,
                                    patch_size=4, num_classes=1000,
                                    ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
                                    ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
                                    ssm_init="v0", forward_type="v05_noz",
                                    mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
                                    patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
                                    downsample_version="v3", patchembed_version="v2",
                                    use_checkpoint=False, posembed=False, imgsize=512)

        self.skip_in = ConvInAct(input_nc, dims, 3, 1, 1)

        fpn_dim = dims * 8
        self.ppm = PyramidPoolingModule(fpn_dim, fpn_dim)
        self.fpn = FeaturePyramidNet(dims, fpn_dim)
        self.fuse = ConvInAct(fpn_dim * 4, fpn_dim, 1, 1, 0)
        self.seg = nn.Sequential(ConvInAct(fpn_dim, fpn_dim, 1, 1, 0),
                                 nn.Conv2d(fpn_dim, output_nc, 1, 1, 0, bias=True))
        self.out = nn.Conv2d(output_nc + dims, output_nc, 3, 1, 1)

    def forward(self, x):
        feat_1, feat_2, feat_3, feat_4 = self.vmamba(x)

        ppm = self.ppm(feat_4)

        fpn_1, fpn_2, fpn_3, fpn_4 = self.fpn([feat_1, feat_2, feat_3, ppm])

        out_size = fpn_1.shape[2:]
        list_f = [fpn_1, F.interpolate(fpn_2, out_size, mode='bilinear', align_corners=False),
                  F.interpolate(fpn_3, out_size, mode='bilinear', align_corners=False),
                  F.interpolate(fpn_4, out_size, mode='bilinear', align_corners=False)]
        feat_out = self.seg(self.fuse(torch.cat(list_f, dim=1)))
        feat_out = F.interpolate(feat_out, x.shape[2:], mode='bilinear', align_corners=False)

        pred = self.out(torch.concat([feat_out, self.skip_in(x)], dim=1))

        return pred
