import torch 
from torch import nn
import glob
import numpy as np
import codecs

import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
from thop import profile
#parameter counter 
def params_counter(model,*input):
    flops, params = profile(model, inputs=input)
    return flops,params

station2idx = {}
with codecs.open("DATA/monitorij_cn27_prov.txt",'r',"utf-8") as f:
    for i in f.readlines():
        line = i.rstrip().split('\t')
        station2idx[line[0]]=(int(line[-2]),int(line[-1]))
        
        


shift_Greenwich = 8
month = 10
for month,hours in [[1,744],[2,672],[4,720],[7,744],[10,744]]:
    obs_label = -999*np.ones((hours+shift_Greenwich,6,182,232))
    
    for filename in glob.glob('DATA/obs2015_{}/*.txt'.format(month)):
        #print(filename)
        station = filename.split('/')[-1].split('.')[0]
        if station not in station2idx:
            print(station,'not in station 2 idx')
            continue
        with codecs.open(filename,'r') as f:    
            for idx,line in enumerate(f.readlines()):
                his = obs_label[idx,:,station2idx[station][1],station2idx[station][0]]
                cur = [float(k) for k in line.split()[1:]]
                obs_label[idx,:,station2idx[station][1],station2idx[station][0]] = [cur[j] if cur[j]!=-999 else his[j] for j in range(len(cur))]
    
    obs_label = obs_label[shift_Greenwich:]
    print('DATA/obs2015_{}_{}'.format( month,'_'.join([str(i) for i in obs_label.shape])))
    np.save('DATA/obs2015_{}_{}'.format( month,'_'.join([str(i) for i in obs_label.shape])),obs_label) 
    
    
#show figure
#136,115 北京， 139,112,天津    130,107石家庄    157,76宁波  96,99兰州    112,91西安
for a,b in [[136,115],[139,112],[130,107],[157,76],[96,99],[112,91]]:
#     plt.plot(label[:,a,b][obs[:,a,b]>0],'b--')
    plt.plot(obs_label[:,3,a,b][obs_label[:,1,a,b]>0],'r--')
    plt.legend(['CTM','obs'])
    plt.show()