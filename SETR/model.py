import torch
import torch.nn as nn

from attention_layer import *

class Encoder(nn.Module):
    '''
    Input:
        - img: batch of images from cityscapes, shape (N, 3, 1024, 2048)
    
    Output:
        - feature: reshaped image, shape (N, embed_dim, 64, 128)
    
    '''
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.pos_vec = nn.Parameter(torch.zeros([1, 8192, embed_dim]))
        self.unfold = nn.Unfold(kernel_size=(16, 16), stride=(16, 16))
        self.linear = nn.Linear(768, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(d_model=embed_dim, n_head=4, n_layers=4, d_hid=embed_dim, dropout=0.1)
        self.reshape = nn.Fold(output_size=(64, 128), kernel_size=1)
    
    def forward(self, img):
        patch_vec = self.linear(self.unfold(img).transpose(1, 2))  # N * 8192 * C
        feature, attn = self.transformer(patch_vec + self.pos_vec)

        return self.reshape(feature.transpose(1, 2)), attn

class NaiveDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(NaiveDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 19, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=16)
        )
        
    def forward(self, feature):
        return self.upsample(feature)
        
class PupDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(PupDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
            nn.Conv2d(64, 19, kernel_size=1),
            nn.BatchNorm2d(19),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
,        )
       
    def forward(self, feature):
        return self.upsample(feature)
        
class Net(nn.Module):
    def __init__(self, embed_dim):
        self.encoder = Encoder(embed_dim)
        self.decoder = NaiveDecoder(embed_dim)
        
    def forward(self, img):
        feature, _ = self.encoder(img)
        return decoder(feature)