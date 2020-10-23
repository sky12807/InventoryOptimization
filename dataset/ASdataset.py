import gc
import glob
import bisect
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from util import simplify_matrix

class AS_Data(Dataset):
    def __init__(self,cfg,left = 0,right = 1,window=24,pollution = ['PM25','O3']):
        super(AS_Data,self).__init__()
        
        
        self.bucket = [0]
        self.window = window
        self.EM = []
        self.METCRO2D = []
        self.METCRO3D = []
        self.METCRO3D_5height = []
        self.grid = np.load(glob.glob(cfg['grid'])[0])
        self.label = []
        
        self.pollution = pollution
        self.pollution_idx_dic = {"NO2":0,"SO2":1,"O3":2,"PM25":3,"PM10":4,"CO":5}
        self.pollution_idx = np.array([self.pollution_idx_dic[i] for i in pollution])
        
        for filename in sorted(glob.glob(cfg['label'])):
            print(filename+'   is loading')
            label = np.load(filename)
            tick = label.shape[0]
            self.label.append(label[int(left*tick):int(right*tick),self.pollution_idx].astype(np.float32).copy())
            self.bucket.append(self.bucket[-1]+int(right*tick)-int(left*tick)-window+1)
            del label
            
        print(self.label[0].shape)
        _,_,W,H = self.label[0].shape
        self.W,self.H = W,H
        
        
        l_EM = [[0,2,3,4,8,9,10,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],range(11,30),[6,7],[30,31],[1],[5]]        
        l_METCRO2D = [[0],[3],[5],[11],[12],[13],[14],[17],[18],[19],[20],[26]]

        for filename in sorted(glob.glob(cfg['METCRO3D_5height'])):
            print(filename+'   is loading')
            METCRO3D_5height = np.load(filename)
            tick,_,W,H = METCRO3D_5height.shape
            self.METCRO3D_5height.append(METCRO3D_5height[int(left*tick):int(right*tick)].astype(np.float32).copy())
            del METCRO3D_5height

            
        for filename in sorted(glob.glob(cfg['EM'])):
            print(filename+'   is loading')
            EM = np.load(filename)
#             EM = simplify_matrix(EM,1,l_EM)
            tick,_,W,H = EM.shape
            self.EM.append(EM[int(left*tick):int(right*tick)].astype(np.float32).copy())
            del EM
            
        for filename in sorted(glob.glob(cfg['METCRO2D'])):
            print(filename+'   is loading')
            METCRO2D = np.load(filename)
#             METCRO2D = simplify_matrix(METCRO2D,1,l_METCRO2D)
            tick,_,W,H = METCRO2D.shape
            self.METCRO2D.append(METCRO2D[int(left*tick):int(right*tick)].astype(np.float32).copy())
            del METCRO2D
            
        for filename in sorted(glob.glob(cfg['METCRO3D'])):
            print(filename+'   is loading')
            METCRO3D = np.load(filename)
            tick,_,W,H = METCRO3D.shape
            self.METCRO3D.append(METCRO3D[int(left*tick):int(right*tick)].astype(np.float32).copy())
            del METCRO3D
            
        
        
    def __getitem__(self,idx):
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
        
#         mi = np.min(d,axis = (2,3),keepdims = True)
#         ma = np.max(d,axis = (2,3),keepdims = True)
#         m = (mi+ma)/2
#         lower = d.copy()
#         higher = d.copy()
#         m = np.tile(m,[1,1,lower.shape[2],lower.shape[3]])
#         lower[lower>=m] = m[lower>=m]
#         higher[higher<=m] = m[higher<=m]
#         d = np.concatenate([lower,higher],axis = 1)

        #### model re-train
        return d,self.grid,self.label[bucket_idx][cur-self.window],self.label[bucket_idx][cur-1]
        
    def __len__(self):
        return self.bucket[-1] - 1