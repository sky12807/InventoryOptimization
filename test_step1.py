import sys
sys.path.append( '/anaconda/envs/py37_pytorch/lib/python3.7')
sys.path.append('/anaconda/envs/py37_pytorch/lib/python3.7/site-packages')

import matplotlib
import matplotlib.pyplot as plt

import yaml
import glob
import torch
import numpy as np
from torch import nn
from torchvision.models import ResNet
from torch.utils.data import DataLoader,Dataset

import logging
from importlib import reload  # Not needed in Python 2

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

from dataset.ASdataset import AS_Data
from dataset.ASdataset_obs_train_input import AS_Data_obs

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

# test_model = res8(51+34,27,inplanes=64,layers = [2],T=T,pre_dim = 2) #+5*16
# name = 'res'
test_model = UNet(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,bilinear=False,pre_dim = len(pollution)) #+80
name = cfg['name']
# test_model.load_state_dict(torch.load('model_save/o3_best_unet2_1month_65_epoch.t'))

test_model.to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(test_model.parameters(),lr=1e-3)

def score(model,loader,criterion= nn.L1Loss(),percent = False):
    model.eval()
    ls = []
    for idx,i in enumerate(loader):
        with torch.no_grad():
            input,grid,yt_1,label = i
            input,grid,yt_1,label = input.to(device),grid.to(device),yt_1.to(device),label.to(device)
            y_pred = model(input,grid,yt_1)
            
            cur_loss = []
            for j in range(label.shape[1]):
                if percent:
                    for esp in [0.1,1,4,8,12,16]:
                        loss = torch.mean(torch.abs(y_pred[:,j]-label[:,j])/(label[:,j]+esp))
                        cur_loss.append(loss.cpu().data)
                else:
                    loss = criterion(y_pred[:,j],label[:,j])
                    cur_loss.append(loss.cpu().data)

            ls.append(cur_loss)
            
    return np.mean(np.array(ls),axis = 0)

best_score = 1000
early_stop = 15
early_cnt = 0
for epoch in range(21):
    logging.info('-----------{}-----------'.format(epoch))
    ls = []
    
    test_model.train()
    for idx,i in enumerate(trainloader):
        print(idx)
        input,grid,yt_1,label = i
        input,grid,yt_1,label = input.to(device),grid.to(device),yt_1.to(device),label.to(device)
        y_pred = test_model(input,grid,yt_1)
        
        assert y_pred.shape == label.shape
        
        optimizer.zero_grad()
        
        loss = criterion(y_pred,label)
        loss.backward()
        optimizer.step()
        ls.append(loss.cpu().data)
        if len(ls)%40==0:
            print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    
    print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    test_score_L1 = score(test_model,testloader,criterion = nn.L1Loss()) 
    print('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))
    
#     if epoch%5 == 0:
#         torch.save(test_model.cpu().state_dict(),'model_save/{}_{}_epoch.t'.format(name,epoch))
#         test_model.to(device)
    if np.sum(test_score_L1)<best_score:
        early_cnt = 0
        best_score = np.sum(test_score_L1)
        torch.save(test_model.cpu().state_dict(),name)
        test_model.to(device)
    else:
        early_cnt += 1
        if early_cnt>=early_stop:
            break