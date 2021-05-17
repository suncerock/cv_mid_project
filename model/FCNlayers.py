import torch
import torch.nn as nn
import torchvision


class _FCNBackbone(nn.Module):
    def __init__(self):
        super(_FCNBackbone, self).__init__()
        
    def forward(self, X):
        '''
        Input: 512 * 512
        Output: 
            - feature32: 16 * 16
            - feature16: 32 * 32
            - feature8: 64 * 64
        '''
        feature8 = self.downsample8(X)
        feature16 = self.downsample16(feature8)
        feature32 = self.downsample32(feature16)
        return feature32, feature16, feature8

class AlexNetBackbone(_FCNBackbone):
    def __init__(self, pretrained=False):
        super(AlexNetBackbone, self).__init__()
        features = torchvision.models.alexnet(pretrained=pretrained).features        
        self.downsample8 = nn.Sequential(
            features[:3],
            nn.AdaptiveAvgPool2d(output_size=(64, 64))
        )
        self.downsample16 = nn.Sequential(
            features[3:6],
            nn.AdaptiveAvgPool2d(output_size=(32, 32))
        )
        self.downsample32 = nn.Sequential(
            features[6:],
            nn.AdaptiveAvgPool2d(output_size=(16, 16))
        )       
    
class VGGBackbone(_FCNBackbone):
    def __init__(self, pretrained=False):
        super(VGGBackbone, self).__init__()
        features = torchvision.models.vgg16_bn(pretrained=pretrained).features
        self.downsample8 = features[:24]
        self.downsample16 = features[24:34]
        self.downsample32 = features[34:]
        
class ResNetBackbone(_FCNBackbone):
    def __init__(self, pretrained=False):
        super(ResNetBackbone, self).__init__()
        ResNet = torchvision.models.resnet18(pretrained=pretrained)
        self.downsample8 = nn.Sequential(
            ResNet.conv1,
            ResNet.bn1,
            ResNet.relu,
            ResNet.maxpool,
            ResNet.layer1,
            ResNet.layer2
        )
        self.downsample16 = ResNet.layer3
        self.downsample32 = ResNet.layer4

    
class Decoder32x(nn.Module):
    def __init__(self, feature32_dim, feature16_dim, feature8_dim, num_classes):
        super(Decoder32x, self).__init__()
        self.conv32 = nn.Conv2d(feature32_dim, num_classes, kernel_size=1, padding=0)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32, padding=0)
        
    def forward(self, feature32, feature16, feature8):
        return self.upsample(self.conv32(feature32))
    
class Decoder16x(nn.Module):
    def __init__(self, feature32_dim, feature16_dim, feature8_dim, num_classes):
        super(Decoder16x, self).__init__()
        self.conv32 = nn.Conv2d(feature32_dim, num_classes, kernel_size=1, padding=0)
        self.conv16 = nn.Conv2d(feature16_dim, num_classes, kernel_size=1, padding=0)
        self.upsample16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, padding=0)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=16, padding=0)
        
    def forward(self, feature32, feature16, feature8):
        return self.upsample(self.conv16(feature16) + self.upsample16(self.conv32(feature32)))
    
class Decoder8x(nn.Module):
    def __init__(self, feature32_dim, feature16_dim, feature8_dim, num_classes):
        super(Decoder8x, self).__init__()
        self.conv8 = nn.Conv2d(feature8_dim, num_classes, kernel_size=1, padding=0)
        self.conv16 = nn.Conv2d(feature16_dim, num_classes, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(feature32_dim, num_classes, kernel_size=1, padding=0)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, padding=0)
        self.upsample16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, padding=0)
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, padding=0)
        
    def forward(self, feature32, feature16, feature8):
        return self.upsample(self.upsample8(self.conv16(feature16) + self.upsample16(self.conv32(feature32))) + self.conv8(feature8))