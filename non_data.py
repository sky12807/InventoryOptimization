import numpy as np
import netCDF4 as nc
import glob
from copy import deepcopy
import matplotlib.pyplot as plt

grid = []
with open('REGID.txt','r',encoding='utf-8') as f:
    for line in f:
        grid.append(line.strip())

nation = np.zeros((182,232))
for i in range(182):
    for j in range(232):
        nation[i][j] = int(grid[i*232+j])

foreign = []
for i in range(182):
    for j in range(232):
        if nation[i][j] == 0:
            foreign.append((i,j))

data = []
for i in range(1,10):
    data.append(nc.Dataset('/scratch/tmp/EM_copy_new/EM_201500{}'.format(i), 'a'))
for i in range(10,32):
    data.append(nc.Dataset('/scratch/tmp/EM_copy_new/EM_20150{}'.format(i), 'a'))
for i in range(91,100):
    data.append(nc.Dataset('/scratch/tmp/EM_copy_new/EM_20150{}'.format(i), 'a'))
for i in range(100,122):
    data.append(nc.Dataset('/scratch/tmp/EM_copy_new/EM_2015{}'.format(i), 'a'))
for i in range(182,213):
    data.append(nc.Dataset('/scratch/tmp/EM_copy_new/EM_2015{}'.format(i), 'a'))
for i in range(274,305):
    data.append(nc.Dataset('/scratch/tmp/EM_copy/EM_2015{}'.format(i), 'a'))
print(len(data))
print(type(data[0]))

var_keys = [name for name in data[0].variables]
var_keys = var_keys[1:]
print(type(var_keys))
print(var_keys)
print(len(var_keys))

count = 0
for d in data:
    variable = d.variables
    count += 1
    print(f'Process {count}/{len(data)}')
    for em in var_keys:
        variable[em] = variable[em] * nation
        # em_data = variable[em]
        # for f in foreign:
            # em_data[:,:,f[0],f[1]] = 0
        # print(f'EM: {em}')
