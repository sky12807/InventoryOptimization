import gc
import glob
import bisect
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from util import simplify_matrix
from dataset.ASdataset import AS_Data

class AS_Data_obs(AS_Data):
    def __init__(self,cfg,left = 0,right = 1,window=24,EM_idx = np.arange(51),pollution = ['PM25','O3']):
        print(pollution)
        super(AS_Data_obs,self).__init__(cfg,left,right,window,EM_idx,pollution=pollution)
        
        self.obs_label = []
        self.finetune_label = []
        
        ####"NO2","SO2","O3","PM2.5","PM10","CO"  need to set
        #### NO2 ppb, SO2 ppb, O3 ppb, PM25 ugm3, PM10 ugm3, CO ppb
        self.obs_label_idx = self.pollution_idx #2 if 'O3' in cfg['label'] else 3
        for filename in sorted(glob.glob(cfg['obs_label'])):
            print(filename+'   is loading')
            obs_label = np.load(filename)
            temp = obs_label.copy()
            ####TRANSPOSE NO2  SO2 O3 CO ppb  ->  ugm3
            obs_label[:,0] = obs_label[:,0] * 22.4/46
            obs_label[:,1] = obs_label[:,1] * 22.4/64
            obs_label[:,2] = obs_label[:,2] * 22.4/48
            obs_label[:,5] = obs_label[:,5] * 22.4/28
            #bug fix about no observation!!!!!
            obs_label[temp==-999] = -999
            
            tick = obs_label.shape[0]
            self.obs_label.append(obs_label[int(left*tick):int(right*tick),self.obs_label_idx].astype(np.float32).copy())
            self.finetune_label.append(obs_label[int(left*tick):int(right*tick),self.obs_label_idx].astype(np.float32).copy())
            del obs_label
        
        self.EM_idx = EM_idx
        self.EM_base = [i.copy() for i in self.EM]
        
    def __getitem__(self,index):
        
        idx = index
        bucket_idx = bisect.bisect_right(self.bucket,idx)-1
        idx -= self.bucket[bucket_idx]
        cur = idx+self.window

        em = self.EM[bucket_idx][idx:cur]
        metcro2d = self.METCRO2D[bucket_idx][idx:cur]
#         metcro3d = self.METCRO3D[bucket_idx][idx:cur]
#         metcro3d_5height = self.METCRO3D_5height[bucket_idx][idx:cur]
        
        
        #metcro3d ,grid 
        d = np.concatenate([em,metcro2d],axis = 1) #metcro3d metcro3d_5height
           
        ### please pay attention!!!! use [0:6] feature , we should forecast current res , current time stamp is 6-1 
        ## output: T*dim*182*232
        return index,d,self.grid,self.label[bucket_idx][cur-self.window],self.label[bucket_idx][cur-1],self.obs_label[bucket_idx][cur-1]
        
        
    
    def update(self,indexes,ds,final=True):
        for i,idx in enumerate(indexes):            
            bucket_idx = bisect.bisect_right(self.bucket,idx)-1
            idx -= self.bucket[bucket_idx]
            cur = idx+self.window
            
            ###update your input
            cur_inventory = ds[i][:,self.EM_idx].cpu().numpy()
            
            self.EM[bucket_idx][idx:cur,self.EM_idx] = cur_inventory
            ### the input of inventory must be positive!!!!!
            if final == True: self.EM[bucket_idx][idx:cur] = np.clip(self.EM[bucket_idx][idx:cur],a_min = 0,a_max = 2*self.EM_base[bucket_idx][idx:cur])

#             self.METCRO2D[bucket_idx][idx:cur] = ds[i][:,51:].cpu().numpy()
            
    def update_labels(self,indexes,labels):
        for i,idx in enumerate(indexes):            
            bucket_idx = bisect.bisect_right(self.bucket,idx)-1
            idx -= self.bucket[bucket_idx]
            cur = idx+self.window
            
            ###### label can't be negative!!!!!
            cur_label = labels[i].cpu().detach().numpy()
            self.finetune_label[bucket_idx][cur-1][cur_label>0] = cur_label[cur_label>0]
    
    def __len__(self):
        return self.bucket[-1] - 1
    
def main():    
    cfg = {'EM':'/AS_data/Emis_npy/EM_2015_07*',
            'label':'/AS_data/Conc_npy/PM25_2015_07*',
            'grid':'/AS_data/Grid_npy/grid_27_182_232.npy',
            'METCRO2D':'/AS_data/METCRO2D_npy/METCRO2D_2015_07*',
            'METCRO3D':'',
            'METCRO3D_5height':'',
            'obs_label':'/AS_data/obs_npy/obs2015_7_*'}

    print('train data is loading ')
    Data = AS_Data_obs(cfg,left = 0,right = 0.3,window = 6)
    trainloader = DataLoader(Data,batch_size=4,shuffle=True)

    ###test update
    for i,line in enumerate(trainloader):
        Data.update(line[0],i*torch.ones_like(line[1]))
        break
    