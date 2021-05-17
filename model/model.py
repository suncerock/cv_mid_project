import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from FCNlayers import *
from SETRlayers import *
from DeeplabLayers import *
        
from collections import OrderedDict

class FCNet(nn.Module):
    def __init__(self, config):
        super(FCNet, self).__init__()
        
        self._all_backbone = ['ResNet', 'AlexNet', 'VGGNet']
        if config.backbone == 'ResNet':
            self.backbone = ResNetBackbone(pretrained=False)
        elif config.backbone == 'AlexNet':
            self.backbone = AlexNetBackbone(pretrained=False)
        elif config.backbone == 'VGGNet':
            self.backbone = VGGBackbone(pretrained=False)
        else:
            raise ValueError('backbone must be one of ', self._all_backbone)
        
        if config.pretrained_backbone:
            self.backbone.load_state_dict(torch.load('../pretrained_backbone/' + config.backbone))
        
        self._all_decoder = ['8x', '16x', '32x']
        if config.decoder == '8x':
            self.decoder = Decoder8x(*config.decoder_c)
        elif config.decoder == '16x':
            self.decoder = Decoder16x(*config.decoder_c)
        elif config.decoder == '32x':
            self.decoder = Decoder32x(*config.decoder_c)
        else:
            raise ValueError('decoder must be one of ', self._all_decoder)
    
    def forward(self, img_input):
        feature32, feature16, feature8 = self.backbone(img_input)
        return self.decoder(feature32, feature16, feature8)
    
    def forward(self, img_input):
        feature32, feature16, feature8 = self.backbone(img_input)
        return self.decoder(feature32, feature16, feature8)
 

class SETR(nn.Module):
    def __init__(self, config):
        super(SETR, self).__init__()
        
        self.embed_dim = config.embed_dim
        self.num_classes = config.num_classes
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        
        self.embedding = SETREmbedding(self.embed_dim)
        self.transformer = SETRTransformer(self.embed_dim, self.nhead, self.num_layers)
        
        self._all_decoder = ['naive', 'pup']
        if config.decoder == 'naive':
            self.decoder = SETRNaiveDecoder(self.embed_dim, self.num_classes)
        elif config.decoder == 'pup':
            self.decoder = SETRPupDecoder(self.embed_dim, self.num_classes)
        else:
            raise ValueError('decoder must be one of ', self._all_decoder)
        
    def forward(self, img_input):
        f = self.embedding(img_input)
        f = self.transformer(f)
        return self.decoder(f)

    
class DeepLabV1(nn.Sequential):
    def __init__(self, num_classes):
        super(DeepLabV1, self).__init__()
        
        self.layer1 = _Stem(64)
        self.layer2 = _ResLayer(2, 64, 64, 1, 1)
        self.layer3 = _ResLayer(2, 64, 128, 2, 1)
        self.layer4 = _ResLayer(2, 128, 256, 1, 2)
        self.layer5 = _ResLayer(2, 256, 512, 1, 4)
        self.fc = nn.Conv2d(512, num_classes, 1)

        
class DeepLabV2(nn.Sequential):
    def __init__(self, num_classes):
        super(DeepLabV2, self).__init__()
        self.layer1 = _Stem(64)
        self.layer2 = _ResLayer(2, 64, 64, 1, 1)
        self.layer3 = _ResLayer(2, 64, 128, 2, 1)
        self.layer4 = _ResLayer(2, 128, 256, 1, 2)
        self.layer5 = _ResLayer(2, 256, 512, 1, 4)
        self.aspp = _ASPPSum(512, num_classes, [6, 8, 12, 24])

        
class DeepLabV3(nn.Sequential):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()
        
        self.layer1 = _Stem(64)
        self.layer2 = _ResLayer(2, 64, 64, 1, 1)
        self.layer3 = _ResLayer(2, 64, 128, 2, 1)
        self.layer4 = _ResLayer(2, 128, 256, 2, 1)
        self.layer5 = _ResLayer(3, 256, 512, 1, 2,
                               multi_grids=[1, 2, 4])
        
        self.aspp = _ASPPCat(512, 256, [6, 12, 18])
        
        self.fc1 = _ConvBnReLU(256 * 5, 256, 1, 1, 0, 1)
        self.fc2 = nn.Conv2d(256, num_classes, 1)


class DeepLabV3p(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3p, self).__init__()
        
        self.layer1 = _Stem(64)
        self.layer2 = _ResLayer(2, 64, 64, 1, 1)
        self.layer3 = _ResLayer(2, 64, 128, 2, 1)
        self.layer4 = _ResLayer(2, 128, 256, 2, 1)
        self.layer5 = _ResLayer(3, 256, 512, 1, 2,
                               multi_grids=[1, 2, 4])
        
        self.aspp = _ASPPCat(512, 256, [6, 12, 18])
        
        self.fc1 = _ConvBnReLU(256 * 5, 256, 1, 1, 0, 1)
        
        self.reduce = _ConvBnReLU(64, 48, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(OrderedDict(
            conv1=_ConvBnReLU(256+48, 256, 3, 1, 1, 1),
            conv2=_ConvBnReLU(256, 256, 3, 1, 1, 1),
            conv3=nn.Conv2d(256, num_classes, 1)
        ))
        
    def forward(self, x):
        h = self.layer2(self.layer1(x))
        h_ = self.reduce(h)
        h = self.layer5(self.layer4(self.layer3(h)))
        h = self.fc1(self.aspp(h))
        h = F.interpolate(h, size=h_.shape[2:], mode='bilinear', align_corners=False)
        h = self.fc2(torch.cat([h, h_], dim=1))
        h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        return h