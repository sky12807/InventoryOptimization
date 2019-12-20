#process METBDY2D

all_data = []
for day in range(1,32):
    day = str(day)    
    if len(day)==1:day = '0'+day
    print(day)
    data = Nio.open_file("/AS_data/Grid/METCRO2D_cn27_20150{}.nc".format(day),"r")
    data_input = []
    for i in data.variables:
        if i == 'TFLAG':continue
        now = data.variables[i][:]
        data_input.append(now)
    data_input = np.concatenate(data_input,axis = 1)
    data_input = np.expand_dims(data_input, axis=0)

    all_data.append(data_input)
    
a = np.concatenate(all_data,axis = 0)
a = a[:,:24]
np.save('data/METCRO2D.npy',a)