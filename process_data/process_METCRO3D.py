all_data = []
month = '04'
for day in range(91,121):
    day = str(day)
    if len(day)==1:day = '00'+day
    if len(day)==2:day = '0'+day
    print(day)
    data_input = []
    data = Nio.open_file("/AS_data/Grid/METCRO3D_cn27_2015{}.nc".format(day),"r")
    for i in data.variables:
        if i == 'TFLAG':continue
        now = data.variables[i][:]
        data_input.append(now)

    data_input = np.stack(data_input)
    data_input = np.sum(data_input,axis = 2)
    data_input = np.transpose(data_input, (1, 0,2,3))
    all_data.append(data_input)
a = np.stack(all_data)


a = a[:,:24]
print(a.shape)
T = a.shape[0]*a.shape[1]
a = a.reshape(T,-1,182,232)

s = '_'.join([str(i) for i in a.shape])
print(s)

np.save('/AS_data/METCRO3D_npy/METCRO3D_2015_{}_{}.npy'.format(month,s),a)