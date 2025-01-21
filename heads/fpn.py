import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
class BasicBlock(nn.Module):
    """2 Layer No Expansion Block
    """
    expansion: int = 1
    def __init__(self, c1, c2, s=1, downsample= None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    """3 Layer 4x Expansion Block
    """
    expansion: int = 4
    def __init__(self, c1, c2, s=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return F.relu(out)


resnet_settings = {
    '18': [BasicBlock, [2, 2, 2, 2]],
    '34': [BasicBlock, [3, 4, 6, 3]],
    '50': [Bottleneck, [3, 4, 6, 3]],
    '101': [Bottleneck, [3, 4, 23, 3]],
    '152': [Bottleneck, [3, 8, 36, 3]]
}


class ResNet(nn.Module):
    def __init__(self, model_name: str = '50', in_channels: int = 3) -> None:
        super().__init__()
        assert model_name in resnet_settings.keys(), f"ResNet model name should be in {list(resnet_settings.keys())}"
        block, depths = resnet_settings[model_name]

        self.inplanes = 64
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, depths[0], s=1)
        self.layer2 = self._make_layer(block, 128, depths[1], s=2)
        self.layer3 = self._make_layer(block, 256, depths[2], s=2)
        self.layer4 = self._make_layer(block, 512, depths[3], s=2)


    def _make_layer(self, block, planes, depth, s=1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers


    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))   # [1, 64, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [1, 512/2048, H/32, W/32]
        return x1, x2, x3, x4

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out

class FPNDecoder(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        decoder_out = []

        for i in range(1, len(features)):
            # out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
            decoder_out.append(out)

        out = self.conv_seg(self.dropout(out))
        return out

class FPN(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNet('50')
        self.head = FPNDecoder(in_channels, channel, num_classes)

    def forward(self, x) -> Tensor:
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

class FPN_Dove(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNet('50')
        # print(self.backbone)
        self.backbone.conv1 =  nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.head = FPNDecoder(in_channels, channel, num_classes)

    def forward(self, x) -> Tensor:
        h_rs = np.zeros((8, 8, 512, 512))
        features = self.backbone(x)
        # for i in features:
        #     print(i.shape)
        out = self.head(features)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class FPN_Fuse(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone_hrs = ResNet('50')
        self.backbone = ResNet('50')
        # print(self.backbone)
        self.backbone.conv1 =  nn.Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.head_hrs = FPNDecoder(in_channels, channel, 64)
        self.head = FPNDecoder(in_channels, channel, 64)
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
    # from .backbones.resnet import ResNet
    # backbone = ResNet('50')
    # head = FPNDecoder([256, 512, 1024, 2048], 128, 2)
    x = torch.randn(2, 3, 512, 512)
    # features = backbone(x)
    # out = head(features)
    # out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    out = FPN_Dove([256, 512, 1024, 2048], 128, 4)(x)
    print(out.shape)
