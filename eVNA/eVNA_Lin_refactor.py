import numpy as np
#import freud
from scipy.spatial import Voronoi,voronoi_plot_2d
#from scipy.spatial.distance import cdist
#from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt
import time
from netCDF4 import Dataset
import os


def eVNA(obs,CTM,show = False):
    vor = Voronoi(lonlatcolrow[:,4:],qhull_options='Qbb Qc Qx')
    region_neighbor = [[] for _ in range(len(lonlatcolrow))]
    for i,j in vor.ridge_points:
        region_neighbor[i].append(j)
        region_neighbor[j].append(i)

    
    #to get the grid/cell cluster
    w = 232
    h = 182
    cell_cluster = -1*np.ones((232,182),dtype=np.int32)
    cell_pos = []
    for col in range(232):
        for row in range(182):
            cell_cluster[col,row] = np.argmin(np.sum((np.array([-3118.5+27*col,-2443.5+27*row])-lonlatcolrow[:,4:])**2,axis = -1))
            cell_pos.append([-3118.5+27*col,-2443.5+27*row])
    cell_pos = np.array(cell_pos)
    
    #get the region neighbor matrix
    max_neighbor = max([len(i) for i in region_neighbor])
    regions_sign = np.zeros((len(region_neighbor),max_neighbor),)
    for i in range(len(region_neighbor)):
        _l = len(region_neighbor[i])
        region_neighbor[i].extend([0]*(max_neighbor-_l))
#         region_neighbor[i] = region_neighbor[i]+[0]*(max_neighbor-_l)
        regions_sign[i,:_l] = 1
    region_neighbor  = np.array(region_neighbor)
    print(region_neighbor.shape)
    
    #get the cell's  region neighbor 
    cell_neighbor = region_neighbor[cell_cluster.reshape(-1)]
    print(cell_neighbor.shape)
    
    # get weight matrix
    weight = np.expand_dims(cell_pos,axis = 1)-lonlatcolrow[:,4:][cell_neighbor]
    weight = np.power(np.sum(weight**2,axis = -1),0.5) #distance
    weight = 1/weight**2
    weight = weight*regions_sign[cell_cluster.reshape(-1)]
    
    weight = np.expand_dims(weight,axis = 2)
    weight = weight*(obs[cell_neighbor]!=0)
    weight = weight/np.sum(weight,axis = 1,keepdims=True)
    print(weight.shape)
#     return weight
    
    #get the region's cell
    region_cell = np.array(lonlatcolrow[:,2]*182 + lonlatcolrow[:,3],np.int32) #[[] for _ in range(len(lonlatcolrow))]
    
    res2 = np.sum(weight*obs[cell_neighbor]/CTM[region_cell[cell_neighbor]],axis = 1)*CTM
    res = np.sum(weight*obs[cell_neighbor],axis = 1)

    #res is VNA
    #res2 is eVNA
    return res2






#input begin with 1
varname = ['NO2','SO2','O3','PM2.5','PM10','CO']
varname2 = ['','','','PM25_TOT','','']
var = 4 
year = 2017
mon = 1
#---------

print(time.ctime())

print("read data")
lonlatcolrow = np.load('sitelonlatcolrowxy.npy')  # obs_number*6
ddata = np.load(str(year)+'_'+str(mon)+'_dailyobs.npy') #obs_number*day*air pollution
sim = Dataset(varname2[var-1]+"_"+str(year*100+mon)+".nc","r") #{variable0:day*182*232, variable1:day*182*232,  variable2:day*182*232 .....}
simdata = sim.variables[varname2[var-1]][:] #day*182*232



obs = ddata[:,:,3] #use the varname index 3 means PM2.5
obs[obs==-999] = 0
CTM = simdata.transpose(2,1,0).reshape(232*182,-1)


res = eVNA(obs,CTM)
print(res.shape)
res = res.reshape(232,182,-1)

