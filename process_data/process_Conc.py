import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
import numpy,sys,os
import numpy as np

for month in ['01','02','04','07','10']:
    print("file is reading now : /AS_data/Conc_tshift/CCTM_cb6r3_ae6.COMBINE_ACONC_v52_cn27_2015{}.tshift".format(month))
    data = nc.Dataset("/AS_data/Conc_tshift/CCTM_cb6r3_ae6.COMBINE_ACONC_v52_cn27_2015{}.tshift".format(month),"r")

    stack =[]
    for i in ["NO2","SO2","O3","PM25_TOT","PM10","CO"]:
        stack.append(data.variables[i][8:])

    stack = np.concatenate(stack,axis = 1)
    stack = stack.data
    sha ='_'.join(["NO2","SO2","O3","PM25","PM10","CO"])+'__'+'_'.join([str(i) for i in stack.shape])
    
    print('/AS_data/Conc_npy/TOTAl_2015_{}_{}.npy'.format(month,sha))
    np.save('/AS_data/Conc_npy/TOTAl_2015_{}_{}.npy'.format(month,sha),stack)
