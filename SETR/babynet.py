import torch.nn as nn

class BabyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.layer = nn.ModuleList(
        #     nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=2),
        #     nn.ReLU()
        # )
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        return x