import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Aggregation(nn.Module):
  def __init__(self,input_dim,output_dim, k = 2, bias = True):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.weight = nn.parameter.Parameter(torch.Tensor(input_dim, k * output_dim))
    self.k = k
    if bias:
      self.bias = nn.parameter.Parameter(torch.Tensor(output_dim,1))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
  def reset_parameters(self):
      stdv = 1. / math.sqrt(self.output_dim * self.k)
      self.weight.data.uniform_(-stdv, stdv)
      if self.bias is not None:
          self.bias.data.uniform_(-stdv, stdv)
  def forward(self, x, S):
      z = x.permute(1,0)
      for k in range(1, self.k):
          x = x.permute(1,0) # [batch size, #agents, features] -> [batchsize, features, #agents]
          x = torch.matmul(x, S)
          z = torch.cat((z, x), dim=0)
      y = torch.matmul(z.permute(1,0),self.weight.permute(1, 0))
      # y = nn.ReLU(inplace = True)(y)
      return y

class PaperGNN(nn.Module):
    def __init__(self, input_dim, output_dim, k=2, bias=True):
        super().__init__()
        self.agg1 = Aggregation(input_dim, output_dim, k, bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.agg2 = Aggregation(input_dim, output_dim, k, bias)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, S):
        out = self.agg1(x, S)
        out = self.relu1(out)
        out = self.agg2(out, S)
        out = self.relu2(out)
        return out