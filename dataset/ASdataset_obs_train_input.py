import gc
import glob
import bisect
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from util import simplify_matrix
from dataset.ASdataset import AS_Data

class AS_Data_obs(AS_Data):
    def __init__(self,cfg,left = 0,right = 1,window=24):
        super(AS_Data_obs,self).__init__(cfg,left,right,window)
        
        _,W,H = self.label[0].shape
        self.W,self.H = W,H
        
        self.obs_label = []
        ####"NO2","SO2","O3","PM2.5","PM10","CO"  need to set
        self.obs_label_idx = 2 if 'O3' in cfg['label'] else 3
        for filename in sorted(glob.glob(cfg['obs_label'])):
            print(filename+'   is loading')
            obs_label = np.load(filename)
            tick,_,W,H = obs_label.shape
            self.obs_label.append(obs_label[int(left*tick):int(right*tick),self.obs_label_idx].copy())
            del obs_label
        
    def __getitem__(self,index):
        
        idx = index
        bucket_idx = bisect.bisect_right(self.bucket,idx)-1
        idx -= self.bucket[bucket_idx]
        cur = idx+self.window

        em = self.EM[bucket_idx][idx:cur]
        metcro2d = self.METCRO2D[bucket_idx][idx:cur]
#         metcro3d = self.METCRO3D[bucket_idx][idx:cur]
#         metcro3d_5height = self.METCRO3D_5height[bucket_idx][idx:cur]
        
        grid = np.repeat(self.grid, self.window, axis=0)
        grid = grid.reshape((self.window,-1,self.W,self.H))
        
        #metcro3d ,grid 
        d = np.concatenate([em,metcro2d],axis = 1) #metcro3d metcro3d_5height
   
        return index,d,self.grid,self.label[bucket_idx][cur-self.window],self.label[bucket_idx][cur],self.obs_label[bucket_idx][cur]
        
    
    def update(self,indexes,ds):
        for i,idx in enumerate(indexes):            
            bucket_idx = bisect.bisect_right(self.bucket,idx)-1
            idx -= self.bucket[bucket_idx]
            cur = idx+self.window
            
            ###update your input
            self.EM[bucket_idx][idx:cur] = ds[i][:,:51].cpu().numpy()
            self.METCRO2D[bucket_idx][idx:cur] = ds[i][:,51:].cpu().numpy()
            
        
    
    def __len__(self):
        return self.bucket[-1]
    
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
    