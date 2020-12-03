""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock,conv1x1,conv3x3

from model.unet_parts import *
from model.layers import Attention

class UNet(nn.Module):
    def __init__(self, n_channels,grid_dim, T, pre_dim = 1,bilinear=True):
        super(UNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self.conv_grid = nn.Sequential(
            norm_layer(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=7, stride=1, padding=3,
                               bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
        
        self.bn0 = norm_layer(n_channels)
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.h0 = nn.Parameter(torch.randn(1,1,64))
        self.c0 = nn.Parameter(torch.randn(1,1,64))
        self.rnn = nn.LSTM(n_channels,64,num_layers=1,batch_first=True)
        
        self.inc = DoubleConv(64, 64)
    
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.conv_y_s = nn.ModuleList()
        self.out_conv_s = nn.ModuleList()
        self.pre_dim = pre_dim
        for _ in range(pre_dim):
            self.conv_y_s.append(nn.Sequential(
                                nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
                                                   bias=False),
                                nn.LeakyReLU(),
                                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                nn.LeakyReLU(),
                                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
                                                   bias=False),
                                                   )
                        )
            
            self.out_conv_s.append(nn.Sequential(nn.Conv2d(64+8+8, 64, kernel_size=1), #(8+8+8,64),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(64, 1, kernel_size=1))
                                  )
        
        
        
        
    def forward(self, x,grid,yt_1):
        
        #x B*T*C*H*W
        B,T,C,H,W = x.shape
#         print(f'Initial x shape: {x.shape}')
#         print(f'Grid shape: {grid.shape}')
#         print(f'yt shape: {yt_1.shape}')
        x = x.reshape(B*T,C,H,W)
#         print(f'Reshape x shape: {x.shape}')
#         x = self.bn0(x)
#         print(f'Batch Norm shape: {x.shape}')
        x = x.reshape(B,T,C,H,W)
#         print(f'Reshape x shape: {x.shape}')
        
        x = x.permute(0,3,4,1,2) #x B*H*W*T*C
#         print(f'Permute x shape: {x.shape}')
        x = x.reshape(B*H*W,T,C)
#         print(f'Reshape x shape: {x.shape}')
        h0 = self.h0.repeat(1,B*H*W,1)
        c0 = self.c0.repeat(1,B*H*W,1)
#         print(f'h0 shape: {h0.shape}')
#         print(f'c0 shape: {c0.shape}')
        _,(hn,cn) = self.rnn(x,(h0,c0)) #hn num_layers*(B*H*W)*rnn_hidden
#         print(f'hn shape: {hn.shape}')
#         print(f'cn shape: {cn.shape}')
        x = hn[-1]
#         print(f'x shape: {x.shape}')
        x = x.reshape(B,H,W,-1)
#         print(f'Reshape x shape: {x.shape}')
        x = x.permute(0,3,1,2)
#         print(f'Permute x shape: {x.shape}')
        x = x.reshape(B,-1,H,W)
#         print(f'Reshape x shape: {x.shape}')
        
        x1 = self.inc(x)
#         print(f'After Double Conv: {x1.shape}')
        x2 = self.down1(x1)
#         print(f'After Down 1: {x2.shape}')
        x3 = self.down2(x2)
#         print(f'After Down 2: {x2.shape}')
        x = self.up3(x3, x2)
#         print(f'After up 3: {x.shape}')
        x = self.up4(x, x1)
#         print(f'After up 4: {x.shape}')
        
        logits = x.reshape(B,-1,H,W)
#         print(f'Reshape x to logits: {logits.shape}')
        
        grid = self.conv_grid(grid)
#         print(f'Grid: {grid.shape}')
        
        #yt_1:B*pre_dim*H*W        
        logits = torch.cat([logits,grid],dim = 1)
#         print(f'Concat logits and grid: {logits.shape}')
        out = []
        for air in range(self.pre_dim):
            yt_1_now = self.conv_y_s[air](yt_1[:,air:air+1])
#             print(f'new yt shape: {yt_1_now.shape}')
            out.append(self.out_conv_s[air](torch.cat([logits,yt_1_now],dim = 1)))
#             print(f'part out shape: {out[-1].shape}')
#             out.append(self.out_conv_s[air](logits))
            
        out = torch.cat(out,dim = 1)
#         print(f'out shape: {out.shape}')
        return out
    

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
        self.inc = nn.Sequential(
            BasicBlock(64, 64),
            # BasicBlock(64, 64)
        )
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 512)
        # self.up1 = Up(1024, 256, bilinear)
        # self.up2 = Up(512, 128, bilinear)
        # self.up3 = Up(256, 64, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
                
        self.outt = nn.Sequential(OutConv(64, 1))
        
        
        # nn.Linear(T,1))
        #                           nn.ReLU(),
        #                           nn.Linear(64,1))
        
        
    def forward(self, x,grid):
        x = self.bn0(x)
        x = self.conv0(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
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