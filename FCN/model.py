import torch
import torch.nn as nn
import torchvision

from .layers import *

class FCN_ResNet(nn.Module):
    def __init__(self):
        super(FCN_ResNet, self).__init__()
        self.backbone = ResNetBackbone(pretrained=False)
        self.decoder = Decoder8x(512, 256, 128, 19)
        
    def forward(self, img_input):
        # img_input : (N, 3, 512, 1024)
        feature32, feature16, feature8 = self.backbone(img_input)
        return self.decoder(feature32, feature16, feature8)
    
class FCN_AlexNet(nn.Module):
    def __init__(self):
        super(FCN_AlexNet, self).__init__()
        self.backbone = AlexNetBackbone(pretrained=False)
        self.decoder = Decoder8x(256, 192, 64, 19)
        
    def forward(self, img_input):
        # img_input : (N, 3, 512, 1024)
        feature32, feature16, feature8 = self.backbone(img_input)
        return self.decoder(feature32, feature16, feature8)
    
class FCN_VGGNet(nn.Module):
    def __init__(self):
        super(FCN_VGGNet, self).__init__()
        self.backbone = VGGBackbone(pretrained=False)
        self.decoder = Decoder8x(512, 512, 256, 19)
        
    def forward(self, img_input):
        # img_input : (N, 3, 512, 1024)
        feature32, feature16, feature8 = self.backbone(img_input)
        return self.decoder(feature32, feature16, feature8)