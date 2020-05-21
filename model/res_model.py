import torch
from torch import nn
from torchvision.models.resnet import BasicBlock,conv1x1,conv3x3

class res8(nn.Module):
    def __init__(self,in_channels,grid_dim,inplanes=64,layers=[2],T=24,block=BasicBlock,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(res8,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        grid_output = 16
        self.conv_grid = nn.Sequential(
            norm_layer(grid_dim),
            nn.Conv2d(grid_dim, 32, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
                               )
        
#         self.conv_y = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,
#                                bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
#                                bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         )
        
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.bn0 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.conv_temporal = nn.Conv2d(self.inplanes*T, self.inplanes, kernel_size=1, stride=1,bias=True)
        
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        
        em_dim = 64
        self.layer1_end = nn.Sequential(nn.Conv2d(self.inplanes, em_dim, kernel_size=1, stride=1),
                                        nn.ReLU(inplace=True)
                                       )
        
        self.conv_fc = nn.Sequential(nn.Conv2d(grid_output+em_dim, 64, kernel_size=1, stride=1),
                                     nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, stride=1))
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def forward(self, x,grid,yt_1):
        B,T,C,H,W = x.shape
        
        #x B*T*C*H*W
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.bn0(x)
        x = self.conv1(x)
        
        x = x.reshape(B,-1,H,W) #B*C1*H*W
        x = self.conv_temporal(x) #B*C1*H*W  -> B*C2*H*W
        
        x = self.bn1(x)
        x = self.relu(x)
                
        x = self.layer1(x)
        x = self.layer1_end(x)

        
#         x = x.permute(2,3,0,1)  # [T,C,W,H]  ->  [W,H,T,C]
#         x = x.view(W,H,-1)
        
        grid = self.conv_grid(grid)
#         grid = grid.permute(2,3,0,1)
#         grid = grid.view(W,H,-1)
        
#         yt_1 = self.conv_y(yt_1)
        
        x = torch.cat([x,grid],dim = 1)
        
#         x = x.permute(0,2,3,1)

#         x = x.view(T,W*H,C)
#         h1_n,(hn,cn) = self.rnn(x)
# #         hn : rnn_layer w*h hidden
#         hn = hn.permute(1,0,2).contiguous()
#         hn = hn.view(W,H,-1)
#         x = self.fc(hn)
        x = self.conv_fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    
def test():
    B,T,C,H,W = 3,8,70,100,200
    model=res8(70,27,inplanes=64,layers=[2],T=T)
    x = torch.randn(B,T,C,H,W)
    grid = torch.randn(B,27,H,W)
    yt_1 = None
    
    output = model(x,grid,yt_1)
    print(output.shape)