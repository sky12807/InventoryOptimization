import glob

import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from util import simplify_matrix

class AS_Data(Dataset):
    def __init__(self,cfg,left = 0,right = 1,window=24):
        super(AS_Data,self).__init__()
        
        

        self.EM = []
        self.METCRO2D = []
        self.METCRO3D = []
        self.grid = np.load(glob.glob(cfg['grid'])[0])
        self.label = []
    

        l_EM = [[0,2,3,4,8,9,10,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],range(11,30),[6,7],[30,31],[1],[5]]        
        l_METCRO2D = [[0],[3],[5],[11],[12],[13],[14],[17],[18],[19],[20],[26]]

        for filename in sorted(glob.glob(cfg['EM'])):
            print(filename+'   is loading')
            EM = np.load(filename)
#             EM = simplify_matrix(EM,1,l_EM)
            tick,_,W,H = EM.shape
            self.EM.append(EM[int(left*tick):int(right*tick)])
        self.EM = np.concatenate(self.EM,axis = 0)


        for filename in sorted(glob.glob(cfg['METCRO2D'])):
            print(filename+'   is loading')
            METCRO2D = np.load(filename)
#             METCRO2D = simplify_matrix(METCRO2D,1,l_METCRO2D)
            tick,_,W,H = METCRO2D.shape
            self.METCRO2D.append(METCRO2D[int(left*tick):int(right*tick)])
        self.METCRO2D = np.concatenate(self.METCRO2D,axis = 0)


        for filename in sorted(glob.glob(cfg['METCRO3D'])):
            print(filename+'   is loading')
            METCRO3D = np.load(filename)
            tick,_,W,H = METCRO3D.shape
            self.METCRO3D.append(METCRO3D[int(left*tick):int(right*tick)])
        self.METCRO3D = np.concatenate(self.METCRO3D,axis = 0)


        for filename in sorted(glob.glob(cfg['label'])):
            print(filename+'   is loading')
            label = np.load(filename)
            tick,W,H = label.shape
            self.label.append(label[int(left*tick):int(right*tick)])
        self.label = np.concatenate(self.label,axis = 0)


        _,W,H = self.label.shape
        self.W,self.H = W,H

        self.window = window
        
    def __getitem__(self,idx):
        cur = idx+self.window
        em = self.EM[idx:cur]
        metcro2d = self.METCRO2D[idx:cur]
        metcro3d = self.METCRO3D[idx:cur]
#         grid = np.repeat(self.grid, self.window, axis=0)
#         grid = grid.reshape((self.window,-1,self.W,self.H))
        
        #metcro3d ,grid 
        d = np.concatenate([em,metcro2d,metcro3d],axis = 1)
        return d,self.grid,self.label[cur]
        
    def __len__(self):
        return self.label.shape[0]-self.window