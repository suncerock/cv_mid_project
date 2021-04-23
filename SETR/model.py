import torch
import torch.nn as nn

from SETR.attention_layer import TransformerEncoder

class Encoder(nn.Module):
    '''
    Input:
        - img: batch of images from cityscapes, shape (N, 3, 1024, 2048)
    
    Output:
        - feature: reshaped image, shape (N, embed_dim, 64, 128)
    
    '''
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.pos_vec = nn.Parameter(torch.zeros([1, 2048, embed_dim]))
        self.unfold = nn.Unfold(kernel_size=(32, 32), stride=(32, 32))
        self.linear = nn.Linear(3072, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.transformer = TransformerEncoder(d_model=embed_dim, n_head=1, n_layers=2, d_hid=embed_dim, dropout=0.1)
        self.reshape = nn.Fold(output_size=(32, 64), kernel_size=1)
    
    def forward(self, img):
        patch_vec = self.linear(self.unfold(img).transpose(1, 2))  # N * 3072 * C
        feature = self.transformer(patch_vec + self.pos_vec)

        return self.reshape(feature.transpose(1, 2))

class NaiveDecoder(nn.Module):
    def __init__(self, embed_dim, num_class):
        super(NaiveDecoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_class, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=32)
        )
        
    def forward(self, feature):
        return self.upsample(feature)
        
class PupDecoder(nn.Module):
    def __init__(self, embed_dim, num_class):
        super(PupDecoder, self).__init__()
        self.upsample1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(64, num_class, kernel_size=1),
            nn.BatchNorm2d(19),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
       
    def forward(self, feature):
        return self.upsample3(self.upsample2(self.upsample1(feature)))
        
class Net(nn.Module):
    def __init__(self, embed_dim, num_class):
        super(Net, self).__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = PupDecoder(embed_dim, num_class)
        
    def forward(self, img):
        feature = self.encoder(img)
        return self.decoder(feature)