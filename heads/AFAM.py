import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
# from cbam import CBAM
from .fpn import ResNet
# from .cbam import CBAM
import torch
from torch import nn
from torch.nn.parameter import Parameter

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            # nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # print('o1, o2, mask', o1.shape, o2.shape, mask.shape)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        # print('x, offset, mask', x.shape, offset.shape, mask.shape)
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.csa_layer = csa_layer(c2, 1)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        # feat = torch.mul(x, atten)
        x = self.conv(x)
        feat = self.csa_layer(x)
        x = x + feat
        return  x # self.conv(x)

class FSM_CVPR(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        # self.csa_layer = csa_layer(c2, 1)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return  self.conv(x)

class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        # feat_align = self.dcpack_l2(feat_up, offset)
        return feat_align + feat_arm

class FAM_CVPR(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM_CVPR(c1, c2)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        # feat_align = self.dcpack_l2(feat_up, offset)
        return feat_align + feat_arm

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
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


class csa_layer(nn.Module):
    def __init__(self, channel, num_layers, k_size=3):
        super(csa_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)  # padding=(k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channel,
            nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # self.mlp = torch.nn.Linear(256, 256)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 256)
        # )

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # y_max = self.max_pool(x)
        # print(y.shape)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.transformer_encoder(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_max = self.transformer_encoder(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_local = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(y.shape)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y_local = self.sigmoid(y_local)
        # print(y.shape)

        return x * y.expand_as(x)   # * y_local.expand_as(x)

class ca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)  # padding=(k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()
        #
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=channel,
        #     nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # self.mlp = torch.nn.Linear(256, 256)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 256)
        # )

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # y_max = self.max_pool(x)
        # print(y.shape)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.transformer_encoder(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_max = self.transformer_encoder(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y_local = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(y.shape)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y_local = self.sigmoid(y_local)
        # print(y.shape)

        return x * y.expand_as(x)   # * y_local.expand_as(x)

class ssa_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ssa_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_h = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_w = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)  # padding=(k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()

        encoder_layer_h = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8)
        self.transformer_encoder_h = nn.TransformerEncoder(encoder_layer_h, num_layers=1)

        encoder_layer_w = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8)
        self.transformer_encoder_w = nn.TransformerEncoder(encoder_layer_w, num_layers=1)
        # self.mlp = torch.nn.Linear(256, 256)
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 256)
        # )

    def forward(self, x):
        # feature descriptor on the global spatial information
        h = x.permute(0, 2, 3, 1)
        w = x.permute(0, 3, 2, 1)
        # print(x.shape)
        h = self.avg_pool_h(h)
        # print(h.shape)
        w = self.avg_pool_w(w)
        # print(w.shape)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        h = self.transformer_encoder_h(h.squeeze(-1).transpose(-1, -2)).unsqueeze(-1)
        # print(h.shape)
        w = self.transformer_encoder_w(w.squeeze(-1).transpose(-1, -2)).unsqueeze(-2)
        # print(w.shape)

        m = torch.bmm(h.squeeze(1), w.squeeze(1)).unsqueeze(1)
        # Multi-scale information fusion
        # print(m.shape)
        m = self.sigmoid(m)
        # print(h.shape)
        # w = self.sigmoid(w)
        # print(w.shape)
        return x * m.expand_as(x)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class FaPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.csa_layer = csa_layer(channel, 1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])

        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        #
        # for feat, align_module in zip(features[1:], self.align_modules[1:]):
        #     out = align_module(feat, out)
        #     # out = output_conv(out)

        # out = self.csa_layer(out) + out  # self.conv_seg(self.csa_layer(out) + out)
        # out = self.conv_seg(out)
        return out

class FAMHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.csa_layer = csa_layer(channel, 1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])

        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        #
        # for feat, align_module in zip(features[1:], self.align_modules[1:]):
        #     out = align_module(feat, out)
        #     # out = output_conv(out)

        # out = self.csa_layer(out) + out  # self.conv_seg(self.csa_layer(out) + out)
        # out = self.conv_seg(out)
        return out

class FaPNHead_CVPR(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[1:]:
            self.align_modules.append(FAM_CVPR(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.csa_layer = csa_layer(channel, 1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])

        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            out = align_module(feat, out)
            out = output_conv(out)
        #
        # for feat, align_module in zip(features[1:], self.align_modules[1:]):
        #     out = align_module(feat, out)
        #     # out = output_conv(out)

        # out = self.conv_seg(self.csa_layer(out) + out)
        out = self.conv_seg(out)
        return out

class ConvReLU(nn.Module):
    """docstring for ConvReLU"""

    def __init__(self, channels: int = 2048, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1,
                 out_channels=None):
        super(ConvReLU, self).__init__()
        if out_channels is None:
            self.out_channels = channels
        else:
            self.out_channels = out_channels

        self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=False)
        # self.bn = nn.GroupNorm(self.out_channels, self.out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels: int = 256, key_channels: int = 128, scale: int = 1):
        super(ObjectAttentionBlock, self).__init__()
        self.key_channels = key_channels
        self.scale = scale
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

        self.to_q = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_k = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_v = ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0)
        self.f_up = ConvReLU(key_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, features, context):
        n, c, h, w = features.shape
        if self.scale == 1:
            features = self.pool(features)

        query = self.to_q(features).contiguous().view(n, -1, c)
        key = self.to_k(context).contiguous().view(n, c, -1)
        value = self.to_v(context).contiguous().view(n, -1, c)

        sim_map = torch.matmul(query, key)
        sim_map *= self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map.squeeze(), value.squeeze()).view(n, c, h, w)

        context = self.f_up(context)
        context = self.up(context)
        return context


class SpatialOCR(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.5):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        self.conv_bn_dropout = nn.Sequential(
            ConvReLU(2 * in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            # nn.Dropout2d(dropout),
        )

    def forward(self, feats, context):
        context = self.object_context_block(feats, context)
        output = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return output


class SpatialGather(nn.Module):
    def __init__(self, n_classes: int = 100, scale: int = 1):
        super(SpatialGather, self).__init__()
        self.cls_num = n_classes
        self.scale = scale

    def forward(self, features, probs):
        n, k, _, _ = probs.shape
        _, c, _, _ = features.shape
        probs = probs.view(n, k, -1)
        features = features.contiguous().view(n, -1, c)
        probs = torch.softmax(self.scale * probs, dim=-1)
        ocr_context = torch.matmul(probs, features)
        return ocr_context.view(n, k, c, 1)

class OCR(nn.Module):
    def __init__(self, channels, n_class):
        super(OCR,self).__init__()
        self.channels = channels
        self.n_class = n_class
        self.object_region_representations = SpatialGather(self.n_class)
        self.object_contextual_representations = SpatialOCR(
            in_channels=self.channels,
            key_channels=self.channels,
            out_channels=self.channels,
            scale=1,
            dropout=0,
        )
        self.soft_object_regions = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(channels, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        # self.pixel_representations = ConvReLU(channels, out_channels=channels, kernel_size=3)
        # self.augmented_representation = nn.Conv2d(self.channels, self.n_class, kernel_size=1, padding=0)
    def forward(self, out):
        # print(out.shape)
        out_aux = self.soft_object_regions(out)
        # features = self.pixel_representations(out)
        context = self.object_region_representations(out, out_aux)
        out = self.object_contextual_representations(out, context)
        # out = self.augmented_representation(features)
        # print(out.shape)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out

class FaPN(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.backbone = ResNet('50')
        self.head = FaPNHead_CVPR(in_channels, channel, num_classes)

    def forward(self, x) -> Tensor:
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
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
    out = FaPN([256, 512, 1024, 2048], 128, 4)(x)
    print(out.shape)
