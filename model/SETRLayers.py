import torch
import torch.nn as nn
import torchvision


class SETREmbedding(nn.Module):
    '''
    Input: 512 * 512
    Output: 256 * C
    '''
    def __init__(self, embed_dim):
        super(SETREmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        self.pos_vec = nn.Parameter(torch.zeros([1, 256, self.embed_dim]))
        self.conv = nn.Conv2d(3, self.embed_dim, kernel_size=32, stride=32, bias=False)
        
    def forward(self, x):
        patch_vec = self.conv(x).view(-1, self.embed_dim, 256).transpose(-1, -2)
        return self.pos_vec + patch_vec
    
class SETRTransformer(nn.Module):
    '''
    Input: 256 * C
    Output: C * 16 * 16
    '''
    def __init__(self, embed_dim, nhead, num_layers):
        super(SETRTransformer, self).__init__()
                
        encoder_layer = nn.TransformerEncoderLayer(self.embed_dim, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        f = self.encoder(x)
        return f.transpose(-1, -2).reshape(-1, self.embed_dim, 16, 16)
    
class SETRNaiveDecoder(nn.Sequential):
    def __init__(self, embed_dim, num_classes):
        super(SETRNaiveDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(embed_dim, num_classes, 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(num_classes, num_classes, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=32)
        
class SETRPupDecoder(nn.Sequential):
    def __init__(self, embed_dim, num_classes):
        super(SETRPupDecoder, self).__init__()
        
        self.upsample1 = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        
        self.upsample2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        
        self.upsample3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.upsample4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        
        self.upsample5 = nn.Sequential(
            nn.Conv2d(32, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, 1)
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        