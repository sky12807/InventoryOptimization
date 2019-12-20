CO_label = None
O3_label = None
PM25_label = None
data = Nio.open_file("data/Conc/CCTM_cb6r3_ae6.COMBINE_ACONC_v52_cn27_201501.tshift","r")

CO = data.variables['CO'][8:].reshape(-1,182,232)
O3 = data.variables['O3'][8:].reshape(-1,182,232)
PM25 = data.variables['PM25_TOT'][8:].reshape(-1,182,232)
sha = '_'.join([str(i) for i in CO.shape])

print(sha)
np.save('/AS_data/Conc_npy/CO_2015_10_{}.npy'.format(sha),CO)
np.save('/AS_data/Conc_npy/O3_2015_10_{}.npy'.format(sha),O3)
np.save('/AS_data/Conc_npy/PM25_2015_10_{}.npy'.format(sha),PM25)