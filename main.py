import glob
import logging
import numpy as np
from importlib import reload  # Not needed in Python 2

import torch
from torchvision.models import ResNet
from torch.utils.data import DataLoader,Dataset


from ASdataset import AS_Data
from model import res8,UNet

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
reload(logging)
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='logging.txt',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s',
                    )

# logging.info('\n\n\n\n\n')
# logging.info('with 3 conv grid to concat\n')
# logging.info('2res block, use simple feature: EM inventory simple,2d ALL,3d ALL\n')
def score(model,loader,criterion):
    model.eval()
    ls = []
    for idx,i in enumerate(loader):
        with torch.no_grad():
            input,grid,label = i
            input,label = torch.squeeze(input,0),torch.squeeze(label,0)
            input,grid,label = input.to(device),grid.to(device),label.to(device)
            y_pred = test_model(input) #,grid)

            y_pred = torch.squeeze(y_pred) #,dim = 2)

            y_pred = y_pred.view(label.shape)
            loss = criterion(y_pred,label)
            ls.append(loss.cpu().data)
        
    return np.mean(np.array(ls))


test_model = UNet(6+34+16,48)
name = 'unet_56fea'
# test_model.load_state_dict(torch.load('model_save/38_epoch.t'))

# test_model = res8(6+34+16,27,[3],T=48)

test_model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(test_model.parameters(),lr=1e-3)


cfg = {'EM':'/AS_data/Emis_npy/EM_2015_*',
      'label':'/AS_data/Conc_npy/O3_2015_*',
      'grid':'/AS_data/Grid_npy/grid_27_182_232.npy',
      'METCRO2D':'/AS_data/METCRO2D_npy/METCRO2D_2015_*',
      'METCRO3D':'/AS_data/METCRO3D_npy/METCRO3D_2015_*'}

print('train data is loading ')
Data = AS_Data(cfg,left = 0,right = 0.8,window = 48)
trainloader = DataLoader(Data,batch_size=1,shuffle=True)

print('test data is loading ')
test_Data = AS_Data(cfg,left = 0.8,right = 1,window = 48)
testloader = DataLoader(test_Data,batch_size=1,shuffle=False)


for epoch in range(120):
    logging.info('-----------{}-----------'.format(epoch))
    ls = []
    
    
    for idx,i in enumerate(trainloader):
        input,grid,label = i
        input,label = torch.squeeze(input,0),torch.squeeze(label,0)
        input,grid,label = input.to(device),grid.to(device),label.to(device)
        y_pred = test_model(input) #,grid)

        y_pred = torch.squeeze(y_pred) #,dim = 2)
        optimizer.zero_grad()
        
        y_pred = y_pred.view(label.shape)
        loss = criterion(y_pred,label)
        loss.backward()
        optimizer.step()
        ls.append(loss.cpu().data)
        if len(ls)%400==0:
            logging.info('epoch {} cur loss {}'.format(epoch,np.mean(ls)))
    
    test_score = score(test_model,testloader,criterion)
    logging.info('cur test loss {}'.format(test_score))
    print(test_score)
    
    if epoch%5 == 0:
        torch.save(test_model.cpu().state_dict(),'model_save/{}_{}_epoch.t'.format(name,epoch))
        test_model.to(device)