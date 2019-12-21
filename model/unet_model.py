""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels,grid_dim, T, bilinear=True):
        super(UNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self.conv_grid = nn.Sequential(
            norm_layer(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 4)
                
        self.outt = nn.Sequential(nn.Linear(4*T+32,128),
                                  nn.ReLU(),
                                  nn.Linear(128,1))
        
        
    def forward(self, x,grid):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        T,_,W,H = logits.shape
        logits = logits.permute(2,3,0,1)
        logits = logits.view(W,H,-1)
        
        grid = self.conv_grid(grid)
        grid = grid.permute(2,3,0,1)
        grid = grid.view(W,H,-1)
        
        logits = torch.cat([logits,grid],dim = -1)
        logits = self.outt(logits)
        return logits