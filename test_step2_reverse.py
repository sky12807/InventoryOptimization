import sys
sys.path.append( '/anaconda/envs/py37_pytorch/lib/python3.7')
sys.path.append('/anaconda/envs/py37_pytorch/lib/python3.7/site-packages')
sys.path.append('/home/v-zeyyan/.local/lib/python3.6/site-packages')

import os

import yaml
import glob
import logging
import numpy as np
from importlib import reload  # Not needed in Python 2

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet
from torch.utils.data import DataLoader,Dataset
import seaborn as sns


from dataset.ASdataset import AS_Data
from dataset.ASdataset_obs_train_input import AS_Data_obs

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
reload(logging)
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='logging.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s',
                    )

import matplotlib
import matplotlib.pyplot as plt


class Focal_loss_regression(nn.Module):
    def __init__(self,max_update=10,_lambda=2,):
        super(Focal_loss_regression,self).__init__()
        self._lambda = _lambda
        max_update = np.power(1/max_update,1/_lambda)
        max_update = 1/max_update
        max_update = 1/(max_update-1)
        self.max_update = max_update
        
    def forward(self,pred,target):
        diff_abs = torch.abs(pred-target)
        diff_max = (1+self.max_update)*torch.max(diff_abs)
#         diff_max.detach()
        rate = torch.pow((1-1/diff_max*diff_abs)**self._lambda,-1)
        diff_abs = rate*diff_abs
        
#         return diff_abs
        return torch.mean(diff_abs)


with open('config/cfg.yaml','r') as f:
    cfg = yaml.load(f)

cfg = {**cfg['step1'],**cfg['share_cfg']}
T = cfg['T']
pollution = cfg['pollution']
batch_size = cfg['batch_size']

print('train data is loading ')
Data = AS_Data(cfg['data_path'],left = cfg['train']['left'],right = cfg['train']['right'],window = T,pollution = pollution)
trainloader = DataLoader(Data,batch_size=batch_size,shuffle=True)
print(len(Data))

print('test data is loading ')
test_Data = AS_Data(cfg['data_path'],left = cfg['test']['left'],right = cfg['test']['right'],window = T,pollution = pollution)
testloader = DataLoader(test_Data,batch_size=batch_size,shuffle=True)
print(len(test_Data))

from model.res_model_LSTM import res8
from model.unet_model_LSTM import UNet
from model.layers import Tensor_Parameter

name = cfg['name']

# name = 'res_2layer_correctdata'
# test_model.load_state_dict(torch.load('model_save/res_2layer_9_epoch.t'))
test_model = UNet(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,bilinear=False,pre_dim = len(pollution)) #+80
# test_model = res8(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,pre_dim = len(pollution))
# t2p = Tensor_Parameter()


test_model.to(device)
# t2p.to(device)
# criterion = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(t2p.parameters(),lr=1)
# optimizer = torch.optim.Adam(t2p.parameters(),lr=1e-1)
test_model.load_state_dict(torch.load(name))
# test_model.load_state_dict(torch.load('model_save/o3_best_unet2_1month_65_epoch.t'))



from model.unet_model_LSTM import ReverseModel
reverse_model = ReverseModel(cfg['meteorological_dim'],cfg['grid_dim'],pre_dim = len(pollution),T=T,em_dim=cfg['emission_dim'])
reverse_model.to(device)
criterion = torch.nn.L1Loss()
reverse_optimizer = torch.optim.Adam(reverse_model.parameters(),lr=1e-2)

print('Train the reverse model')
for epoch in range(25):
    print('-----------{}-----------'.format(epoch))
    ls = []
    direct_ls = []
    reverse_model.train()
    test_model.train()
    count = 0
    for idx,i in enumerate(trainloader):
        input,grid,yt_1,label,next_label, next_metro = i
        em = input[:,:,:cfg['emission_dim'],:,:]
        metro = input[:,:,cfg['emission_dim']:,:,:]

        input,em,metro,grid,yt_1,label,next_label, next_metro = input.to(device),em.to(device),metro.to(device),grid.to(device),yt_1.to(device),label.to(device),next_label.to(device),next_metro.to(device)
        em_pred = reverse_model(next_metro,grid,next_label)
        new_em = torch.cat([em[:,:-1,:,:,:],em_pred.detach()],dim=1)
        x_pred = torch.cat([new_em,metro],dim=2)
        reverse_optimizer.zero_grad()
        label_pred = test_model(x_pred,grid,yt_1) #输入yt_1但是模型中没用
        loss = criterion(label_pred,label)
        loss.backward()
        reverse_optimizer.step()
        direct_loss = criterion(em_pred.detach(),em[:,-1:,:,:,:])
        count += 1
        if count%40 == 0: print(f'Direct Loss: {direct_loss.cpu().data}; F Loss: {loss.cpu().data}')
        ls.append(loss.cpu().data)
        direct_ls.append(direct_loss.cpu().data)
    print(f'Average Training Loss: Direct Loss: {np.mean(np.array(direct_ls))}; F Loss: {np.mean(np.array(ls))}')
    eval_loss = []
    eval_direct_loss = []
    print('********Evaluating*********')
    for idx,i in enumerate(testloader):
        input,grid,yt_1,label,next_label, next_metro = i
        em = input[:,:,:cfg['emission_dim'],:,:]
        metro = input[:,:,cfg['emission_dim']:,:,:]

        input,em,metro,grid,yt_1,label,next_label, next_metro = input.to(device),em.to(device),metro.to(device),grid.to(device),yt_1.to(device),label.to(device),next_label.to(device),next_metro.to(device)
        em_pred = reverse_model(next_metro,grid,next_label)
        new_em = torch.cat([em[:,:-1,:,:,:],em_pred.detach()],dim=1)
        x_pred = torch.cat([new_em,metro],dim=2)
        label_pred = test_model(x_pred,grid,yt_1) #输入yt_1但是模型中没用
        loss = criterion(label_pred,label)
        direct_loss = criterion(em_pred.detach(),em[:,-1:,:,:,:])
        eval_loss.append(loss.cpu().data)
        eval_direct_loss.append(direct_loss.cpu().data)
    print(f'------------Evaluating: Direct Loss: {np.mean(np.array(eval_direct_loss))}; F Loss: {np.mean(np.array(eval_loss))}')
    torch.save(reverse_model.cpu().state_dict(),'model_save/reverse_norm_4month_f_loss.t')
    