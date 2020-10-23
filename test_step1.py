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
reload(logging)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='logging.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s',
                    )



from dataset.ASdataset import AS_Data
from dataset.ASdataset_obs_train_input import AS_Data_obs
import json

# NORM_METHOD = 2

# if NORM_METHOD == 1:
#     with open('Norm1.json', "r") as f:
#         dic = json.load(f)
# elif NORM_METHOD == 2:
#     with open('Norm2.json', "r") as f:
#         dic = json.load(f)

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

# test_model = res8(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,pre_dim = len(pollution)) #+5*16
name = cfg['name']
test_model = UNet(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,bilinear=False,pre_dim = len(pollution)) #+80
# name = cfg['name']
# test_model.load_state_dict(torch.load('model_save/o3_best_unet2_1month_65_epoch.t'))

test_model.to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(test_model.parameters(),lr=1e-3)

def score(model,loader,criterion= nn.L1Loss(),percent = False):
    model.eval()
    ls = []
    for idx,i in enumerate(loader):
        with torch.no_grad():
            input,grid,yt_1,label,all_label= i
            input,grid,yt_1,label,all_label = input.to(device),grid.to(device),yt_1.to(device),label.to(device),all_label.to(device)
            y_pred = model(input,grid,yt_1)
            
#             if NORM_METHOD == 1:
#                 y_pred = y_pred * (dic['max'] - dic['min']) + dic['min']
#                 label = label * (dic['max'] - dic['min']) + dic['min']
#             elif NORM_METHOD == 2:
#                 y_pred = (y_pred + dic['min']) * dic['std'] + dic['mean']
#                 label = (label + dic['min']) * dic['std'] + dic['mean']
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
for epoch in range(25):
#     logging.info('-----------{}-----------'.format(epoch))
    print('-----------{}-----------'.format(epoch))
    ls = []
    
    test_model.train()
    for idx,i in enumerate(trainloader):
#         print(idx)
        input,grid,yt_1,label,all_label= i
        input,grid,yt_1,label,all_label= input.to(device),grid.to(device),yt_1.to(device),label.to(device),all_label.to(device)
        y_pred = test_model(input,grid,yt_1)
#         print('*'*20)
        assert y_pred.shape == label.shape
        
        optimizer.zero_grad()
        
        loss = criterion(y_pred,label)
        loss.backward()
        optimizer.step()
        ls.append(loss.cpu().data)
        if len(ls)%40==0:
            logging.info('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
            print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
#         torch.save(test_model.cpu().state_dict(),'model_save/test.t')
    
#     logging.info('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    test_score_L1 = score(test_model,testloader,criterion = nn.L1Loss()) 
#     logging.info('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))
    print('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))
    
#     if epoch%5 == 0:
#         torch.save(test_model.cpu().state_dict(),'model_save/{}_{}_epoch.t'.format(name,epoch))
#         test_model.to(device)

    if np.sum(test_score_L1)<best_score:
        early_cnt = 0
        best_score = np.sum(test_score_L1)
        torch.save(test_model.cpu().state_dict(),name)
        test_model.to(device)
        print("This is current best epoch")
    # else:
    #     early_cnt += 1
    #     if early_cnt>=early_stop:
    #         break