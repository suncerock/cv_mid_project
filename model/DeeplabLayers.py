import torch
import torch.nn as nn
import torchvision


class _ConvBnReLU(nn.Sequential):
    '''
    A C-B(-R) module
    '''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        
        if relu:
            self.relu = nn.ReLU()
            
class _Stem(nn.Sequential):
    '''
    First conv layer
    7 * 7 kernel with stride=2 and padding=1
    '''
    def __init__(self, c_out):
        super(_Stem, self).__init__()
        self.conv1 = _ConvBnReLU(3, c_out, 7, 2, 3, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)

class _Bottleneck(nn.Module):
    '''
    A basic block in ResNet (can be dilated)
    Reduce - Conv - Increase, with shortcut connection
    A downsampling shortcut is required if stride != 1
    '''
    def __init__(self, c_in, c_out, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        c_hid = c_out // 4
        
        self.reduce = _ConvBnReLU(c_in, c_hid, 1, stride, 0, 1, relu=True)
        self.conv = _ConvBnReLU(c_hid, c_hid, 3, 1, dilation, dilation, relu=True)
        self.increase = _ConvBnReLU(c_hid, c_out, 1, 1, 0, 1, relu=False)
        
        self.shortcut = (
            _ConvBnReLU(c_in, c_out, 1, stride, 0, 1, relu=False)
            if downsample
            else lambda x:x
        )
        
    def forward(self, x):
        h = self.reduce(x)
        h = self.conv(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)
    
class _ResLayer(nn.Sequential):
    '''
    Residual layer
    Several blocks stacked
    First block requires a downsample and a change of channels
    '''
    def __init__(self, n_layers, c_in, c_out, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        
        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)
            
        for i in range(n_layers):
            self.add_module(
                'block{}'.format(i+1),
                _Bottleneck(
                    c_in=c_in if i == 0 else c_out,
                    c_out=c_out,
                    stride=stride if i == 0 else 1,
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False)
                )
            )
  
class _ASPPSum(nn.Module):
    def __init__(self, c_in, c_out, rates):
        super(_ASPPSum, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                'c{}'.format(i),
                nn.Conv2d(c_in, c_out, 3, 1, padding=rate, dilation=rate)
            )
            
    def forward(self, x):
        return sum([state(x) for state in self.children()]) 
            
class _ImagePool(nn.Module):
    def __init__(self, c_in, c_out):
        super(_ImagePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(c_in, c_out, 1, 1, 0, 1)
        
    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode='bilinear', align_corners=False)
        return h
    
class _ASPPCat(nn.Module):
    def __init__(self, c_in, c_out, rates):
        super(_ASPPCat, self).__init__()
        self.c0  = _ConvBnReLU(c_in, c_out, 1, 1, 0, 1)
        for i, rate in enumerate(rates):
            self.add_module(
                'c{}'.format(i+1),
                _ConvBnReLU(c_in, c_out, 3, 1, padding=rate, dilation=rate)
            )
        self.imagepool = _ImagePool(c_in, c_out)
            
    def forward(self, x):
        return torch.cat([state(x) for state in self.children()], dim=1)
                    
