import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperMLP(nn.Module):
  def __init__(self,input_dim,output_dim):
    super().__init__()
    self.mlp = nn.Linear(in_features=input_dim,out_features=output_dim,bias = True)
  def forward(self,x):
    return self.mlp(x)