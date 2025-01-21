import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .fpn import ResNet
import numpy as np

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
        )



class PPM(nn.Module):
    """Pyramid Pooling Module in PSPNet
    """
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(c1, c2, 1)
            )
        for scale in scales])

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []

        for stage in self.stages:
            outs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out



class AlignedModule(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.down_h = nn.Conv2d(c1, c2, 1, bias=False)
        self.down_l = nn.Conv2d(c1, c2, 1, bias=False)
        self.flow_make = nn.Conv2d(c2 * 2, 2, k, 1, 1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor) -> Tensor:
        high_feature_origin = high_feature
        H, W = low_feature.shape[-2:]
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(high_feature, size=(H, W), mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim=1))
        high_feature = self.flow_warp(high_feature_origin, flow, (H, W))
        return high_feature

    def flow_warp(self, x: Tensor, flow: Tensor, size: tuple) -> Tensor:
        # norm = torch.tensor(size).reshape(1, 1, 1, -1)
        norm = torch.tensor([[[[*size]]]]).type_as(x).to(x.device)
        H = torch.linspace(-1.0, 1.0, size[0]).view(-1, 1).repeat(1, size[1])
        W = torch.linspace(-1.0, 1.0, size[1]).repeat(size[0], 1)
        grid = torch.cat((W.unsqueeze(2), H.unsqueeze(2)), dim=2)
        grid = grid.repeat(x.shape[0], 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=False)
        return output


class SFHead(nn.Module):
    def __init__(self, in_channels, channel=256, num_classes=19, scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(in_channels[-1], channel, scales)

        self.fpn_in = nn.ModuleList([])
        self.fpn_out = nn.ModuleList([])
        self.fpn_out_align = nn.ModuleList([])

        for in_ch in in_channels[:-1]:
            self.fpn_in.append(ConvModule(in_ch, channel, 1))
            self.fpn_out.append(ConvModule(channel, channel, 3, 1, 1))
            self.fpn_out_align.append(AlignedModule(channel, channel//2))

        self.bottleneck = ConvModule(len(in_channels) * channel, channel, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channel, num_classes, 1)

    def forward(self, features: list) -> Tensor:
        f = self.ppm(features[-1])
        fpn_features = [f]

        for i in reversed(range(len(features) - 1)):
            feature = self.fpn_in[i](features[i])
            f = feature + self.fpn_out_align[i](feature, f)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()

        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:], mode='bilinear', align_corners=True)

        output = self.bottleneck(torch.cat(fpn_features, dim=1))
        output = self.conv_seg(self.dropout(output))
        return output

class sfnet(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNet('50')
        self.head = SFHead(in_channels, channel, num_classes)

    def forward(self, x) -> Tensor:
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

class sfnet_Dove(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNet('50')
        self.backbone.conv1 =  nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.head = SFHead(in_channels, channel, num_classes)

    def forward(self, x) -> Tensor:
        h_rs = np.zeros((8, 8, 512, 512))
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class sfnet_Fuse(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone_hrs = ResNet('50')
        self.backbone = ResNet('50')
        # print(self.backbone)
        self.backbone.conv1 =  nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.head_hrs = SFHead(in_channels, channel, 64)
        self.head = SFHead(in_channels, channel, 64)
        self.fc = nn.Conv2d(64*2, num_classes, kernel_size=1, padding=0)

    def forward(self, h_rs, x):
        # print(h_rs.shape)
        # h_rs = np.zeros((8, 8, 512, 512))
        features_hrs = self.backbone_hrs(h_rs)
        features = self.backbone(x)
        # for i in features:
        #     print(i.shape)
        out = self.head(features)
        out_hrs = self.head_hrs(features_hrs)
        out = F.interpolate(out, size=out_hrs.shape[-2:], mode='bilinear', align_corners=False)

        fuse = torch.cat([out, out_hrs], 1)

        out = self.fc(fuse)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)

        return out

if __name__ == '__main__':
    import sys
    # sys.path.insert(0, '.')
    # from models.backbones.resnet import ResNet
    # backbone = ResNet('50')
    # for y in backbone:
    #     print(y.shape)
    # head = FaPNHead([256, 512, 1024, 2048], 128, 4)
    x = torch.randn(8, 3, 512, 512)
    # features = backbone(x)
    # print(features.shape)
    # out = head(features)
    # out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    out = sfnet([256, 512, 1024, 2048], 128, 4)(x)
    print(out.shape)
