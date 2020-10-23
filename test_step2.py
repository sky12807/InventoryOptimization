import sys
sys.path.append( '/anaconda/envs/py37_pytorch/lib/python3.7')
sys.path.append('/anaconda/envs/py37_pytorch/lib/python3.7/site-packages')

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


from dataset.ASdataset import AS_Data
from dataset.ASdataset_obs_train_input import AS_Data_obs

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

import matplotlib
import matplotlib.pyplot as plt

voc_idx = np.array([0,2,3,4,8,9,10] + list(range(32,51)))
### grad limit!!!
def hook_fn(bn,input_grad,output_grad):
    # print(module)
    # for g in input_grad: print('input grad shape: {}'.format(g.size()))
    # for g in output_grad: print('output grad shape: {}'.format(g.size()))

    std = bn.running_var

    t_grad = input_grad[0]*torch.sqrt(std + 1e-5).view(-1,1,1)
    mean_grad = torch.mean(t_grad[:,voc_idx,:,:],dim=1,keepdim=True)

    t_grad[:,np.array(voc_idx),:,:] = mean_grad

    x_grad = t_grad/torch.sqrt(std + 1e-5).view(-1,1,1)

    return (x_grad,input_grad[1],input_grad[2])


# logging.info('\n\n\n\n\n')
# logging.info('with 3 conv grid to concat\n')
# logging.info('2res block, use simple feature: EM inventory simple,2d ALL,3d ALL\n')



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

def show_station_diff(month_idx = 0):

    for month in [month_idx]:
        for p in range(len(Data.pollution_idx)):
            p_name = pollution[p]

            month_obs = Data.obs_label[month][24:,p].copy()
            month_nn = Data.finetune_label[month][24:,p].copy()
            month_ctm = Data.label[month][24:,p].copy()

            ### pixel with no obersertion -> 0
            month_ctm[month_obs==-999] = 0
            month_nn[month_obs==-999] = 0
            month_obs[month_obs==-999] = 0

            sum_month_obs = np.sum(month_obs,axis = 0)
            sum_month_nn = np.sum(month_nn,axis = 0)
            sum_month_ctm = np.sum(month_ctm,axis = 0)

            plt.rcParams['figure.figsize'] = (10,5)

            a = ((sum_month_nn - sum_month_ctm)/(sum_month_ctm+1))[sum_month_obs>0]
            plt.hist(a,bins = 100,range = (-1,1))
            plt.title(f'{p_name}: (nn-ctm)/(1+ctm):{np.mean(np.abs(a))}')
            plt.show()


            a = ((sum_month_nn - sum_month_ctm)/(sum_month_obs+1))[sum_month_obs>0]
            plt.hist(a,bins = 100,range = (-1,1))
            plt.title(f'{p_name}: (nn-ctm)/(1+obs):{np.mean(np.abs(a))}')
            plt.show()
            a = ((sum_month_nn - sum_month_obs)/(sum_month_obs+1))[sum_month_obs>0]
            plt.hist(a,bins = 100,range = (-1,1))
            plt.title(f'{p_name}: (nn-obs)/(1+obs):{np.mean(np.abs(a))}')
            plt.show()   
            a = ((sum_month_ctm - sum_month_obs)/(sum_month_obs+1))[sum_month_obs>0]
            plt.hist(a,bins = 100,range = (-1,1))
            plt.title(f'{p_name}: (ctm-obs)/(1+obs):{np.mean(np.abs(a))}')
            plt.show()

            plt.rcParams['figure.figsize'] = (15, 5)

            for b,a,name in [[136,115,'beijing'],[139,112,'tianjin'],[130,107,'shijiazhuang'],[157,76,'ningbo'],[96,99,'lanzhou'],[112,91,'xian'],[48,139,'xinjiang']]:
                obs = month_obs[:,a,b]
                nn = month_nn[:,a,b]
                ctm = month_ctm[:,a,b]

                plt.plot(obs[obs!=-999],'k')
                plt.plot(nn[obs!=-999],'r--')
                plt.plot(ctm[obs!=-999],'g--')

                plt.legend(['obs','nn_label','ctm_label'])
                plt.title(p_name + ' : ' + name)
                plt.show()

with open('config/cfg.yaml','r') as f:
    cfg = yaml.load(f)
    
cfg = {**cfg['step2'],**cfg['share_cfg'],**cfg['share_cfg']['data_path']}
T = cfg['T']
pollution = cfg['pollution']
batch_size = cfg['batch_size']

name = cfg['name']
### remove CH4,AACD,ACET
EM_idx = np.array(cfg['EM_idx'])
EM_save_path = cfg['EM_save_path']

print('train data is loading ')
Data = AS_Data_obs(cfg,left = 0,right = 1,window = T,pollution = pollution,EM_idx = EM_idx)
trainloader = DataLoader(Data,batch_size=batch_size,shuffle=True)
print(len(Data))
print(Data.EM_idx)

from model.res_model_LSTM import res8
from model.unet_model_LSTM import UNet
from model.layers import Tensor_Parameter

# test_model = res8(51+34,27,inplanes=64,layers = [2],T=24) #+5*16
# name = 'res_2layer_correctdata'
# test_model.load_state_dict(torch.load('model_save/res_2layer_9_epoch.t'))
test_model = UNet(cfg['meteorological_dim']+cfg['emission_dim'],cfg['grid_dim'],T=T,bilinear=False,pre_dim = len(pollution)) #+80
t2p = Tensor_Parameter()


test_model.to(device)
t2p.to(device)
criterion = torch.nn.L1Loss()
# optimizer = torch.optim.SGD(t2p.parameters(),lr=1)
optimizer = torch.optim.Adam(t2p.parameters(),lr=1e-1)
test_model.load_state_dict(torch.load(name))
# test_model.load_state_dict(torch.load('model_save/o3_best_unet2_1month_65_epoch.t'))


def score(model,loader,criterion= nn.L1Loss() ):
    model.eval()
    ls = []
    for idx,i in enumerate(loader):
        with torch.no_grad():
            indexes,input,grid,yt_1,label,obs = i[0],i[1],i[2],i[3],i[4],i[5]
            input,grid,yt_1,label,obs = input.to(device),grid.to(device),yt_1.to(device),label.to(device),obs.float().to(device)
            input = t2p(input)
            y_pred = model(input,grid,yt_1)
            
            Data.update_labels(indexes,y_pred)
            if torch.sum(obs!=-999)==0:
                continue
            
            cur_loss = []
            for pollution in range(y_pred.shape[1]):
                cur_pred = y_pred[:,pollution]
                cur_obs = obs[:,pollution]
                loss = criterion(cur_pred[cur_obs!=-999],cur_obs[cur_obs!=-999])
                cur_loss.append(loss.cpu().data)
            ls.append(cur_loss)
        
    return np.mean(np.array(ls),axis = 0)

def hook_NO(bn,input_grad,output_grad):
    # print(module)
    # for g in input_grad: print('input grad shape: {}'.format(g.size()))
    # for g in output_grad: print('output grad shape: {}'.format(g.size()))

    std = bn.running_var

    t_grad = input_grad[0]*torch.sqrt(std + 1e-5).view(-1,1,1)
    mean_grad = torch.mean(t_grad[:,np.array([2,3]),:,:],dim=1,keepdim=True)

    t_grad[:,np.array([2,3]),:,:] = mean_grad

    x_grad = t_grad/torch.sqrt(std + 1e-5).view(-1,1,1)

    return (x_grad,input_grad[1],input_grad[2])

modules = test_model.named_children()
# print(type(modules))
# for name, module in modules:
# #     print(f'{name}:{module}')
#     if name == 'bn0':
#         print(f'Name: {name}')
#         print(f'Module: {module}')
#         module.register_backward_hook(hook_NO)


for epoch in range(25):
#     logging.info('-----------{}-----------'.format(epoch))
    print('-----------{}-----------'.format(epoch))
    ls = []
    if epoch == 24: final = True
    else: final = True
    test_model.eval()
    test_model.rnn.train()
    for idx,i in enumerate(trainloader):
        
        indexes,input,grid,yt_1,label,obs = i[0],i[1],i[2],i[3],i[4],i[5]
        input,grid,yt_1,label,obs = input.to(device),grid.to(device),yt_1.to(device),label.to(device),obs.to(device)
        input = t2p(input)
        y_pred = test_model(input,grid,yt_1)

        if torch.sum(obs!=-999)==0:
            continue
        optimizer.zero_grad()
        
        loss = criterion(y_pred[obs!=-999],obs[obs!=-999])
        loss.backward()
        optimizer.step()
        ls.append(loss.cpu().data)
        
        #udpate input parameter
        Data.update(indexes,t2p.Input.data,final = final)
#         Data.update_labels(indexes,y_pred)
        if len(ls)%100==0:
#             logging.info('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
            print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    print('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
#     logging.info('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    test_score_L1 = score(test_model,trainloader,criterion = nn.L1Loss()) 
#     logging.info('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))
    print('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))
#     if epoch%3 == 0:
#         test_score_L1 = score(test_model,trainloader,criterion = nn.L1Loss()) 
#         logging.info('-------------cur test loss L1:  {}'.format(','.join([str(s) for s in test_score_L1])))

    if epoch%1 == 0:
#         for _idx,month in enumerate(['01','02','04','07','10'][:len(Data.EM)]):
        for _idx,month in enumerate(['10'][:len(Data.EM)]):
            if not os.path.exists(EM_save_path):
                os.mkdir(EM_save_path)
#             print(idx)
            np.save(EM_save_path+'/rest_4month_SULF_CO_7dim_{}_finetune_input.npy'.format(month),Data.EM[_idx])
#             np.save('month_{}_finetune_input.npy'.format(month),Data.EM[_idx])
