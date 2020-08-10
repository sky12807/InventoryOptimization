import numpy as np
import netCDF4 as nc
import glob
from copy import deepcopy


nc.default_fillvals['f4'] = 0
nc.default_fillvals['f8'] = 0

data = []
for i in range(1,10):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_201500{}'.format(i), 'a'))
for i in range(10,32):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_20150{}'.format(i), 'a'))
print(len(data))
print(type(data[0]))

print(data[2].variables['ALD2'][0][0][20])

for filename in sorted(glob.glob('/AS_data/zeyuan_folder/reload_data/adjust_new_7EM_01_744_51_182_232.npy')):
    print(filename)
    final = np.load(filename)
    print(final.shape)
final = np.expand_dims(final,axis = 1)
print(final.shape)

var_keys = [name for name in data[0].variables]
var_keys = var_keys[1:]
print(type(var_keys))
print(var_keys)
print(len(var_keys))

for i in range(len(data)): 
    variable = data[i].variables
    for em in range(51):
        em_data = deepcopy(variable[var_keys[em]][:-1])
        final_em = final[i*24:i*24+24,:,em,:,:]
        sum_em = np.sum(em_data, axis = 1, keepdims = True)
        finetune_em = em_data * final_em / sum_em
        variable[var_keys[em]][:-1] = finetune_em
    print('Process {}/{}'.format(i,len(data)))

print('*'*20)
print('Month 04')
print('*'*20)

data = []
for i in range(91,100):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_20150{}'.format(i), 'a'))
for i in range(100,121):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_2015{}'.format(i), 'a'))
print(len(data))
print(type(data[0]))

for filename in sorted(glob.glob('/AS_data/zeyuan_folder/reload_data/adjust_new_7EM_04_720_51_182_232.npy')):
    print(filename)
    final = np.load(filename)
    print(final.shape)
final = np.expand_dims(final,axis = 1)
print(final.shape)

var_keys = [name for name in data[0].variables]
var_keys = var_keys[1:]
print(type(var_keys))
print(var_keys)
print(len(var_keys))

for i in range(len(data)): 
    variable = data[i].variables
    for em in range(51):
        em_data = deepcopy(variable[var_keys[em]][:-1])
        final_em = final[i*24:i*24+24,:,em,:,:]
        sum_em = np.sum(em_data, axis = 1, keepdims = True)
        finetune_em = em_data * final_em / sum_em
        variable[var_keys[em]][:-1] = finetune_em
    print('Process {}/{}'.format(i,len(data)))

print('*'*20)
print('Month 07')
print('*'*20)

data = []
for i in range(182,213):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_2015{}'.format(i), 'a'))
print(len(data))
print(type(data[0]))

for filename in sorted(glob.glob('/AS_data/zeyuan_folder/reload_data/adjust_new_7EM_07_744_51_182_232.npy')):
    print(filename)
    final = np.load(filename)
    print(final.shape)
final = np.expand_dims(final,axis = 1)
print(final.shape)

var_keys = [name for name in data[0].variables]
var_keys = var_keys[1:]
print(type(var_keys))
print(var_keys)
print(len(var_keys))

for i in range(len(data)): 
    variable = data[i].variables
    for em in range(51):
        em_data = deepcopy(variable[var_keys[em]][:-1])
        final_em = final[i*24:i*24+24,:,em,:,:]
        sum_em = np.sum(em_data, axis = 1, keepdims = True)
        finetune_em = em_data * final_em / sum_em
        variable[var_keys[em]][:-1] = finetune_em
    print('Process {}/{}'.format(i,len(data)))

print('*'*20)
print('Month 10')
print('*'*20)

data = []
for i in range(274,305):
    data.append(nc.Dataset('/scratch/tmp/EM_non/EM_2015{}'.format(i), 'a'))
print(len(data))
print(type(data[0]))

for filename in sorted(glob.glob('/AS_data/zeyuan_folder/reload_data/adjust_new_7EM_10_744_51_182_232.npy')):
    print(filename)
    final = np.load(filename)
    print(final.shape)
final = np.expand_dims(final,axis = 1)
print(final.shape)

var_keys = [name for name in data[0].variables]
var_keys = var_keys[1:]
print(type(var_keys))
print(var_keys)
print(len(var_keys))

for i in range(len(data)): 
    variable = data[i].variables
    for em in range(51):
        em_data = deepcopy(variable[var_keys[em]][:-1])
        final_em = final[i*24:i*24+24,:,em,:,:]
        sum_em = np.sum(em_data, axis = 1, keepdims = True)
        finetune_em = em_data * final_em / sum_em
        variable[var_keys[em]][:-1] = finetune_em
    print('Process {}/{}'.format(i,len(data)))