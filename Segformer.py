import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
# from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import MiT, ResNet, PVTv2, ResNetD
from backbones.layers import trunc_normal_
from heads import SegFormerHead, FaPNHead, SFHead, UPerHead, FPNHead, FaPNCBAMHead
# from model_fuse import CEFEB

from torchvision.ops import DeformConv2d
# from cbam import CBAM
# from .fpn import ResNet
# from .cbam import CBAM
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
# from CBAM import CBAM
# from ACmixAttention import SelfAttinACmix
# from SKAttention import SKAttention
# from CoordAttention import CoordAtt
from backbones.mobilenetv2 import MobileNetV2
# class ConvModule(nn.Sequential):
#     def __init__(self, c1, c2, k, s=1, p=0):
#         super().__init__(
#             nn.Conv2d(c1, c2, k, s, p, bias=False),
#             # nn.BatchNorm2d(c2),
#             nn.ReLU(True)
#         )




class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.act(y)

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)



segformer_settings = {
    'B0': 256,        # head_dim
    'B1': 256,
    'B2': 768,
    'B3': 768,
    'B4': 768,
    'B5': 768
}



class SegFormer(nn.Module):
    def __init__(self, variant: str = 'B0', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = MiT(variant)
        self.decode_head = SegFormerHead(self.backbone.embed_dims, segformer_settings[variant], num_classes)
        # self.decode_head = FaPNHead([64, 128, 320, 512], 128, 19)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        feature = self.backbone(x)
        # for i in feature:
        #     print(i.shape)
        y = self.decode_head(feature)   # 4x reduction in image size
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return feature, y


class Segformer_baseline(nn.Module):
    def __init__(self, n_class):
        super(Segformer_baseline, self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        # self.semantic_img_model = SegFormer('B1', self.out_channels)
        # self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b1.ade.pth', map_location='cpu'))
        # # self.semantic_img_model_2 = SegFormer('B1', self.out_channels)
        # # self.semantic_img_model_2.load_state_dict(torch.load('.\\segformer.b1.ade.pth', map_location='cpu'))
        # # self.semantic_img_model.backbone.patch_embed1.proj = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 256, self.n_class)

        self.backboneViT = MiT('B1')  # MobileNetV2()
        self.backboneViT.load_state_dict(torch.load('mit_b1.pth', map_location='cpu'), strict=False)

        self.decode_head_ViT = SegFormerHead([64, 128, 320, 512], 256, self.n_class)

    def forward(self, h_rs):
        features = self.backboneViT(h_rs)
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        # print(h_rs[:, :, :, :3].shape)
        out = self.decode_head_ViT(features)
        # features2, out2 = self.semantic_img_model_2(h_rs[:, 3:, :, :])

        # fuse = torch.cat([out, out2], 1)
        # out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Semiformer(nn.Module):
    def __init__(self, n_class):
        super(Semiformer, self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.dim = 128

        self.backboneViT = MiT('B1')  # MobileNetV2()
        self.backboneViT.load_state_dict(torch.load('mit_b1.pth', map_location='cpu'), strict=False)

        self.backboneCNN = MobileNetV2()
        self.backboneCNN.load_state_dict(torch.load('mobilenet_v2-b0353104.pth', map_location='cpu'), strict=False)

        self.decode_head_ViT = FaPNCBAMHead([64, 128, 320, 512], 128, self.n_class)
        # self.decode_head_CNN = FaPNHead([24, 32, 96, 320], 128, self.n_class)
        self.decode_head_Fuse = FaPNCBAMHead([64+24, 128+32, 320+96, 512+320], 128, self.n_class)

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        # print(h_rs[:, :, :, :3].shape)
        Vit_out = self.backboneViT(h_rs)
        CNN_out = self.backboneCNN(h_rs)

        fuse_out = (torch.cat([Vit_out[0], CNN_out[0]], 1), torch.cat([Vit_out[1], CNN_out[1]], 1),torch.cat([Vit_out[2], CNN_out[2]], 1),torch.cat([Vit_out[3], CNN_out[3]], 1))

        out_0 = self.decode_head_ViT(Vit_out)

        out = self.decode_head_Fuse(fuse_out)

        # semantic_out = self.decode_path(out)
        # print(out_0.shape, out_1.shape, out.shape)
        #     print(feature.shape)

        out_0 = F.interpolate(out_0, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out_1 = F.interpolate(out_1, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)

        return out_0, out


class Semiformer_student(nn.Module):
    def __init__(self, n_class):
        super(Semiformer_student, self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.dim = 128

        self.backboneViT = MiT('B1')  # MobileNetV2()
        self.backboneViT.load_state_dict(torch.load('mit_b1.pth', map_location='cpu'), strict=False)

        self.backboneCNN = MobileNetV2()
        self.backboneCNN.load_state_dict(torch.load('mobilenet_v2-b0353104.pth', map_location='cpu'), strict=False)

        self.decode_head_ViT = FaPNCBAMHead([64, 128, 320, 512], 128, self.n_class)
        # self.decode_head_CNN = FaPNHead([24, 32, 96, 320], 128, self.n_class)
        self.decode_head_Fuse = FaPNCBAMHead([64+24, 128+32, 320+96, 512+320], 128, self.n_class)

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        # print(h_rs[:, :, :, :3].shape)
        Vit_out = self.backboneViT(h_rs)
        CNN_out = self.backboneCNN(h_rs)

        fuse_out = (torch.cat([Vit_out[0], CNN_out[0]], 1), torch.cat([Vit_out[1], CNN_out[1]], 1),torch.cat([Vit_out[2], CNN_out[2]], 1),torch.cat([Vit_out[3], CNN_out[3]], 1))

        out_0 = self.decode_head_ViT(Vit_out)

        out = self.decode_head_Fuse(fuse_out)

        # semantic_out = self.decode_path(out)
        # print(out_0.shape, out_1.shape, out.shape)
        #     print(feature.shape)

        out_0 = F.interpolate(out_0, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out_1 = F.interpolate(out_1, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)

        return out_0, out

class Semiformer_finetune(nn.Module):
    def __init__(self, n_class):
        super(Semiformer_finetune, self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.dim = 128

        # self.Semiformer_finetune = Semiformer_student(self.n_class+1)
        self.Semiformer_finetune = torch.load('.\\model-UV-Semantic\\UV-Semantic_CJZY-semiformer-student-withmask-all-0.5-0.7#1.pth', map_location='cpu')

        # self.conv_1x1_vit = nn.Conv2d(3, 2, 1, 1, 0)
        # self.conv_1x1_fuse = nn.Conv2d(3, 2, 1, 1, 0)
        # self.backboneViT = MiT('B1')  # MobileNetV2()
        # self.backboneViT.load_state_dict(torch.load('mit_b1.pth', map_location='cpu'), strict=False)
        #
        # self.backboneCNN = MobileNetV2()
        # self.backboneCNN.load_state_dict(torch.load('mobilenet_v2-b0353104.pth', map_location='cpu'), strict=False)
        #
        self.Semiformer_finetune.decode_head_ViT = FaPNCBAMHead([64, 128, 320, 512], 128, self.n_class)
        # # self.decode_head_CNN = FaPNHead([24, 32, 96, 320], 128, self.n_class)
        self.Semiformer_finetune.decode_head_Fuse = FaPNCBAMHead([64+24, 128+32, 320+96, 512+320], 128, self.n_class)

    def forward(self, h_rs):
        out_0, out = self.Semiformer_finetune(h_rs)
        # print(out_0.shape, out.shape)
        return out_0, out
        # features2, out2 = self.semantic_img_model_2(h_rs[:, 3:, :, :])

        # fuse = torch.cat([out, out2], 1)
        # out = self.conv_block_fc(out)
        # print(features.shape)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # return out

class FSM(nn.Module):
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

class Segformer_student(nn.Module):
    def __init__(self, n_class):
        super(Segformer_student, self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.dim = 256
        self.semantic_img_model = torch.load('.\\model-UV-Semantic\\UV-Semantic_CJZY-segformer-1.pth') # SegFormer('B1', self.out_channels)
        # self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b1.ade.pth', map_location='cpu'))
        # print(self.semantic_img_model.decode_head)
        # self.semantic_img_model.decode_head = SegFormerHeadNew([64, 128, 320, 512], self.dim, self.n_class)
        # self.conv_block_fc = nn.Sequential(
        #     # FCViewer(),
        #     # nn.Conv2d(2, 150, kernel_size=(1, 1), stride=(1, 1)),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(2, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        # )

    def forward(self, h_rs):
        # print(h_rs.shape)
        out = self.semantic_img_model(h_rs)
        # out = self.conv_block_fc(out)
        # print(out.shape)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Segformer_b3(nn.Module):
    def __init__(self, n_class):
        super(Segformer_b3, self).__init__()
        self.n_class=n_class
        self.out_channels = 150
        self.semantic_img_model = SegFormer('B3', self.out_channels)
        self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))
        # self.semantic_img_model.decode_head = SegFormerHead([64, 128, 320, 512], 768, self.n_class)
        # self.semantic_img_model.decode_head = UPerHead([64, 128, 320, 512], 128, self.n_class)
        # print(self.semantic_img_model)
        # self.semantic_mmdata_model = SegFormer('B3', self.out_channels)
        # self.semantic_mmdata_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))

        self.conv_block_fc = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(150, 150, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(150, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )
    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        # mmdata = torch.cat([h_rs, mmdata], 1)
        features, out = self.semantic_img_model(h_rs)

        out = self.conv_block_fc(out)
        # print(features.shape)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
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

if __name__ == '__main__':
    FCN = SegFormer_Fuse(4)
    FCN = FCN.to('cuda')
    # summary(model, (3, 512, 512), device="cpu")
    images = torch.randn(2, 3, 1024, 1024)
    images = images.to('cuda')
    # mmdata = mmdata.clone().detach().float()
    images = images.clone().detach().float()
    dove = torch.randn(2, 8, 256, 256)
    dove = dove.to('cuda')
    # mmdata = mmdata.clone().detach().float()
    dove = dove.clone().detach().float()
    out = FCN(images, dove)
    print(out.shape)
