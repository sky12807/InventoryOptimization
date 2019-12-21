import numpy as np

# from thop import profile
# #parameter counter 
# def params_counter(model,*input):
#     flops, params = profile(model, inputs=input)
#     return flops,params


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




def simplify_matrix(mat,axis,list_indices):
    res = []
    for indices in list_indices:
        cur = np.take(mat,np.array(indices),axis = axis)
        if len(indices)>1:
            res.append(np.sum(cur,axis = axis,keepdims = True))
        else:
            res.append(cur)
            
    res = np.concatenate(res,axis = axis)
    return res