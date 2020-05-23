dic = {'ALD2': 0, 'CO': 1, 'ETH': 2, 'FORM': 3, 'ISOP': 4, 'NH3': 5, 'NO': 6, 'NO2': 7, 'UNR': 8, 'OLE': 9, 'PAR': 10, 'PEC': 11, 'PMC': 12, 'PMOTHR': 13, 'PNO3': 14, 'POC': 15, 'PSO4': 16, 'PCL': 17, 'PNH4': 18, 'PNA': 19, 'PMG': 20, 'PK': 21, 'PCA': 22, 'PNCOM': 23, 'PFE': 24, 'PAL': 25, 'PSI': 26, 'PTI': 27, 'PMN': 28, 'PH2O': 29, 'SO2': 30, 'SULF': 31, 'TERP': 32, 'TOL': 33, 'XYL': 34, 'MEOH': 35, 'ETOH': 36, 'ETHA': 37, 'ALDX': 38, 'IOLE': 39, 'CH4': 40, 'AACD': 41, 'NAPH': 42, 'NR': 43, 'SOAALK': 44, 'XYLMN': 45, 'PRPA': 46, 'BENZ': 47, 'ETHY': 48, 'ACET': 49, 'KET': 50}

filenames = ['/AS_data/cn27_cb05/OT_2015001',
           '/AS_data/cn27_cb05/OT_2015032',
           '/AS_data/cn27_cb05/OT_2015091',
           '/AS_data/cn27_cb05/OT_2015182',
           '/AS_data/cn27_cb05/OT_2015274']

emis_filenames = ['/AS_data/Emis_npy/'+i for i in ['EM_2015_01_744_51_182_232.npy',
                                                   'EM_2015_02_672_51_182_232.npy',
                                                   'EM_2015_04_720_51_182_232.npy',
                                                   'EM_2015_07_744_51_182_232.npy',
                                                   'EM_2015_10_744_51_182_232.npy',
                                                  ]]

days = [31,28,30,31,31]
out_filenames = ['/AS_data/Emis_with_foreign_npy/'+i for i in ['EM_2015_01_744_51_182_232.npy',
                                                               'EM_2015_02_672_51_182_232.npy',
                                                               'EM_2015_04_720_51_182_232.npy',
                                                               'EM_2015_07_744_51_182_232.npy',
                                                               'EM_2015_10_744_51_182_232.npy',
                                                              ]]


for i in range(3,len(filenames)):
    filename = filenames[i]
    out_filename = out_filenames[i]
    emis_filename = emis_filenames[i]
    day = days[i]
    print(filename)
    print(out_filename)
    print(emis_filename)
    print(day)
    
    ##load data
    data = nc.Dataset(filename)
    
    foreign = np.zeros((1,24,51,182,232))
    for idx,i in enumerate(data.variables):
        if idx==0:continue

        print(idx,i,dic[i],data.variables[i][:].shape)

        a = np.sum(data.variables[i][:],axis = 1)
        a = a[:24]
        foreign[:,:,dic[i]] = a[:]

    emis = np.load(emis_filename)
    
    emis = emis.reshape(day,24,51,182,232)
    emis = emis + foreign
    emis = emis.reshape(day*24,51,182,232)
    
    np.save(out_filename,emis)
    print('save successfull')