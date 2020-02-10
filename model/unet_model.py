""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock,conv1x1,conv3x3

from model.unet_parts import *
from model.layers import Attention

class UNet(nn.Module):
    def __init__(self, n_channels,grid_dim, T, bilinear=True):
        super(UNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self.conv_grid = nn.Sequential(
            norm_layer(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
        
        self.conv_y = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
        self.bn0 = norm_layer(n_channels)
        self.n_channels = n_channels
        self.bilinear = bilinear
        
#         downsample = nn.Sequential(
#                 conv1x1(n_channels, planes * 1, stride=1),
#                 norm_layer(planes * 1),
#             )

        self.h0 = nn.Parameter(torch.randn(1,1,64))
        self.c0 = nn.Parameter(torch.randn(1,1,64))
        self.rnn = nn.LSTM(n_channels,64,num_layers=1,batch_first=True)
        
        self.inc = DoubleConv(64, 64)
#         sel.inc = nn.Sequential(
#             BasicBlock(n_channels, 64,downsample),
#             BasicBlock(64, 64)
#         )
        
        
        
    
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, 8)
                
        self.outt = nn.Sequential(OutConv(64+8+8, 64), #(8+8+8,64),
                                nn.ReLU(inplace=True),
                                OutConv(64,1)
                               )
        
        
        
        
    def forward(self, x,grid,yt_1):
        
        #x B*T*C*H*W
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.bn0(x)
        x = x.reshape(B,T,C,H,W)
        
        x = x.permute(0,3,4,1,2) #x B*H*W*T*C
        x = x.reshape(B*H*W,T,C)
        h0 = self.h0.repeat(1,B*H*W,1)
        c0 = self.c0.repeat(1,B*H*W,1)
        _,(hn,cn) = self.rnn(x,(h0,c0)) #hn num_layers*(B*H*W)*rnn_hidden
        
        x = hn[-1]
        x = x.reshape(B,H,W,-1)
        x = x.permute(0,3,1,2)
        x = x.reshape(B,-1,H,W)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        
#         logits = self.outc(x) #B*T,_,H,W
        logits = x.reshape(B,-1,H,W)
#         logits = logits.permute(2,3,0,1)
#         logits = logits.view(W,H,-1)
        
        #grid:B*27*H*W
        grid = self.conv_grid(grid)
#         grid = grid.permute(2,3,0,1)
#         grid = grid.view(W,H,-1)
        
        #yt_1:B*1*H*W
        yt_1 = self.conv_y(yt_1)
#         yt_1 = yt_1.permute(2,3,0,1)
#         yt_1 = yt_1.view(W,H,-1)
        
        
        logits = torch.cat([logits,grid,yt_1],dim = 1)
        logits = self.outt(logits)
        return logits
    
    
    

class UNet_Res(nn.Module):
    def __init__(self, n_channels,grid_dim, T, bilinear=True):
        super(UNet_Res, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self.conv_grid = nn.Sequential(
            norm_layer(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
        self.inplanes = 64
        self.bn0 = norm_layer(n_channels)
        self.conv0 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        
        self.bilinear = bilinear
        
#         self.inc = DoubleConv(n_channels, 64)
        self.inc = nn.Sequential(
            BasicBlock(64, 64),
#             BasicBlock(64, 64)
        )
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
                
        self.outt = nn.Sequential(OutConv(64, 1))
        
        
#         nn.Linear(T,1))
#                                   nn.ReLU(),
#                                   nn.Linear(64,1))
        
        
    def forward(self, x,grid):
        x = self.bn0(x)
        x = self.conv0(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        T,_,W,H = logits.shape
        logits = logits.permute(0,2,3,0)
        logits = logits.view(W,H,-1)
        
        grid = self.conv_grid(grid)
        grid = grid.permute(2,3,0,1)
        grid = grid.view(W,H,-1)
        
        logits = torch.cat([logits,grid],dim = -1)
        logits = self.outt(logits)
        return logits