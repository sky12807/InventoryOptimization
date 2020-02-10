import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_features,**kwargs):
        super(Attention, self).__init__()
        self.in_features = in_features

        self.concat = kwargs.get('concat',False)
        self.qk_hidden = kwargs.get('qk_hidden',64) #24
        self.use_residual = kwargs.get('use_residual',True)
        
        self.dropout = nn.Dropout(kwargs.get('dropout',0))
        self.layernorm = nn.LayerNorm(in_features)
        self.Wq = nn.Linear(in_features, self.qk_hidden)
        nn.init.xavier_uniform_(self.Wq.weight, gain=1.414)
        
        self.Wk = nn.Linear(in_features, self.qk_hidden)
        nn.init.xavier_uniform_(self.Wk.weight, gain=1.414)
        
        self.Wr = nn.Parameter(torch.ones(1))
                
        self.temperature = self.qk_hidden**0.5
        
    def forward(self, q,k,v):
        '''
        q: B*1/T*(C*H*W)
        k: B*T*(C*H*W)
        v: B*T*(C*H*W)
        
        return:
        B*1/T*(C*H*W)   shape like q
        '''
        
        r = q
        
        q = self.Wq(q)
        k = self.Wk(k)
        
        r = self.Wr*r
        
        
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = self.dropout(F.softmax(attn, dim=-1))
        res = torch.matmul(attn, v)
                
        if self.use_residual:
            res+=r
        
        if self.concat:
            return F.elu(res)
        else:
            return res