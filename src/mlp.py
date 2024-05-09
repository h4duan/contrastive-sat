import torch
import torch.nn as nn
from torch.nn import LayerNorm
#from torch.nn import BatchNorm 

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, out_dim)


  def forward(self, x):

    #x = self.l1(x)
    x = self.l1(x).relu()
    #print(x.device)
    #x = self.l2(x).relu()
    x = self.l2(x)
    #x = self.l3(x)

    return x

