""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock,conv1x1,conv3x3

from model.unet_parts_wo_BN import *
from model.layers import Attention

    
class ReverseModel(nn.Module):
    def __init__(self, n_channels,grid_dim,pre_dim,T,em_dim,bilinear=True):
        super(ReverseModel, self).__init__()        
#         self.bn_metrod = nn.BatchNorm2d(n_channels)
#         self.bn_y = nn.BatchNorm2d(pre_dim)
#         self.bn_grid = nn.BatchNorm2d(grid_dim)
#         self.bn_y_grid = nn.BatchNorm2d(T)
        self.n_channels = n_channels
        self.rnn = nn.LSTM(n_channels+pre_dim,64,num_layers=1,batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64+8,32),
            nn.LeakyReLU(),
            nn.Linear(32,em_dim)
        )
        
        self.h0 = nn.Parameter(torch.randn(1,1,64))
        self.c0 = nn.Parameter(torch.randn(1,1,64))
        
        self.inc = DoubleConv(64, 64)
    
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.conv_grid = nn.Sequential(
#             nn.BatchNorm2d(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )

    def forward(self,metrod,grid,all_label):
        y_metrod = torch.cat([all_label, metrod],dim = 2)
#         print(f'y_metrod shape: {y_metrod.shape}')
        
        B, T, C, H, W = y_metrod.shape
        y_metrod = y_metrod.permute(0,3,4,1,2)
        y_metrod = y_metrod.reshape(B*H*W,T,C)
#         print(f'reshape y_metrod shape: {y_metrod.shape}')
        
        h0 = self.h0.repeat(1,B*H*W,1)
        c0 = self.c0.repeat(1,B*H*W,1)
#         print(f'h0 shape: {h0.shape}')
#         print(f'c0 shape: {c0.shape}')
        _,(hn,cn) = self.rnn(y_metrod,(h0,c0)) #hn num_layers*(B*H*W)*rnn_hidden
#         print(f'hn shape: {hn.shape}')
#         print(f'cn shape: {cn.shape}')
        y = hn[-1]
#         print(f'y shape: {y.shape}')
        y = y.reshape(B,H,W,-1)
        y = y.permute(0,3,1,2)
#         print(f'reshape y shape: {y.shape}')
        
        y1 = self.inc(y)
#         print(f'After Double Conv: {y1.shape}')
        y2 = self.down1(y1)
#         print(f'After Down 1: {y2.shape}')
        y3 = self.down2(y2)
#         print(f'After Down 2: {y3.shape}')
        y = self.up3(y3, y2)
#         print(f'After up 3: {y.shape}')
        y = self.up4(y, y1)
#         print(f'After up 4: {y.shape}')
        
        logits = y.reshape(B,-1,H,W)
#         print(f'Reshape y to logits: {logits.shape}')
        
        grid = self.conv_grid(grid)
#         print(f'Grid: {grid.shape}')
        
        #Concat logits and grid
        logit_grid = torch.cat([logits,grid],dim=1)
#         print(f'logit_grid shape: {logit_grid.shape}')
        
        logit_grid = logit_grid.permute(0,2,3,1)
#         print(f'logit_grid shape: {logit_grid.shape}')
        out = self.linear(logit_grid)
#         print(f'output shape: {out.shape}')
        out = out.permute(0,3,1,2)
#         print(f'output shape: {out.shape}')
        B,C,H,W = out.shape
        out = out.reshape(B,1,C,H,W)
#         print(f'final output shape: {out.shape}')
        
        return out
    
