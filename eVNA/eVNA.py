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

#input begin with 1
varname = ['NO2','SO2','O3','PM2.5','PM10','CO']
varname2 = ['','','','PM25_TOT','','']
var = 4 
year = 2017
mon = 1
day = 20
#---------

print(time.ctime())

print("read data")
lonlatcolrow = np.load('/data1/home/dingd/dlrsm/emis/obs/day/sitelonlatcolrowxy.npy')
ddata = np.load('/data1/home/dingd/dlrsm/emis/obs/day/'+str(year)+'_'+str(mon)+'_dailyobs.npy')
sim = Dataset("/data1/home/dingd/dlrsm/emis/sim/"+varname2[var-1]+"_"+str(year*100+mon)+".nc","r")
simdata = sim.variables[varname2[var-1]][:]

s = 0
list = []
while(s < lonlatcolrow.shape[0]):
    #tmp = [lonlatcolrow[s,0],lonlatcolrow[s,1]]
    tmp = [lonlatcolrow[s,4],lonlatcolrow[s,5]]
    list.append(tmp)
    s = s + 1

gridvalue = np.zeros((232,182),dtype=np.float)
list.append([0,0])
pnum = len(list)#points.shape[0]

print("calculate")
col = 0
while(col<232):
    row = 0 
    while(row<182):
        modelE = simdata[day-1,row,col]
        grid = [-3118.5+27*col,-2443.5+27*row]
        #print(col,',',row,';',grid)
        list[pnum-1]=grid
        points = np.array(list)
        #plt.scatter(points[:,0], points[:,1])
        #plt.title('Points')
        #plt.xlim((72, 135))
        #plt.ylim((15, 55))
        #plt.show()
        
        #voronoi_kdtree = cKDTree(points)
        #grid = [100,50]
        #test_point_dist, test_point_regions = voronoi_kdtree.query(grid, k=1)
        
        vor = Voronoi(points,qhull_options='Qbb Qc Qx')
        #for i, reg in enumerate(vor.regions):
        #    print('Region:', i)
        #    print('Indices of vertices of Voronoi region:', reg)
        #    print('Associated point:', points[i], '\n')
        #fig = voronoi_plot_2d(vor)
        #fig.show()
        
        #distance = cdist(x1,x2,"euclidean")
        
        reg = vor.point_region[pnum-1]
        vert = vor.regions[reg]
        neighbors = []
        
        r = 0
        for vertices in vor.regions:
            same = 0
            for v in vert:
                if(v in vertices):
                    same = same + 1
                if(same>=2):
                    neighbors.append(r)
            r = r + 1
        
        neighborregions = np.unique(neighbors)
        
        neighborpoint = []
        for r in neighborregions:
            neighborpoint.append(vor.point_region.tolist().index(r))
        
        neighborpoint.remove(pnum-1)
        #print(neighborpoint)
        
        #neighborpointlonlat = np.zeros((len(neighborpoint),2),dtype=np.float)
        weight = np.zeros((len(neighborpoint)),dtype=np.float)
        pdata = np.zeros((len(neighborpoint)),dtype=np.float)
        modeli = np.zeros((len(neighborpoint)),dtype=np.float)
        i = 0
        #print(neighborpoint)
        for p in neighborpoint:
            pdata[i] = ddata[p,day-1,var-1]
            #print(p,day-1,int(lonlatcolrow[p,3]),int(lonlatcolrow[p,2]))
            modeli[i] = simdata[day-1,int(lonlatcolrow[p,3]),int(lonlatcolrow[p,2])]
            x = vor.points[p]
            #neighborpointlonlat[i]=vor.points[p]
            weight[i] = np.power((x[0]-grid[0])**2 +(x[1]-grid[1])**2,1/2)
            i = i + 1
        
        weight = 1/(weight*weight)
        weight = weight/sum(weight)
        pdata[pdata<0] = 0
        gridvalue[col,row] = sum(pdata*modelE/modeli*weight)
        
        #print(gridvalue[col,row])
        row = row + 1 
    col = col + 1
    

print("save")
np.save('eVNA_'+varname[var-1]+'_'+str(year)+'_'+str(mon)+'_d'+str(day)+'.npy',gridvalue)

print(time.ctime())

