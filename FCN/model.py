import torch
import torch.nn as nn
import torchvision

pretrained = False
ResNet = torchvision.models.resnet18(pretrained=pretrained)
#VGG = torchvision.models.vgg16_bn(pretrained=pretrained)

class VGGBackbone(nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        self.backbone = VGG.features

    def forward(self, X):
        return self.backbone(X)

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        self.backbone = nn.Sequential(
            ResNet.conv1,
            ResNet.bn1,
            ResNet.relu,
            ResNet.maxpool,
            ResNet.layer1,
            ResNet.layer2,
            ResNet.layer3,
            ResNet.layer4
        )
        
    def forward(self, X):
        return self.backbone(X)
    
class FCN(nn.Module):
    def __init__(self, num_class):
        super(FCN, self).__init__()
        self.backbone = ResNetBackbone()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_class, kernel_size=2, stride=2)
        )
    def forward(self, img_input):
        # img_input : (N, 3, 512, 1024)
        feature = self.backbone(img_input)
        return self.upsample(feature)