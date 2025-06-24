import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PaperCNN(nn.Module):
  def __init__(self,input_channels):
    super().__init__()
    self.encode = nn.Sequential(
        nn.Conv2d(input_channels,32,kernel_size = 3, stride=1, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size=2),
        #------------------------------------------------------------------
        nn.Conv2d(32,32,kernel_size = 3, stride=1, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True),
        #------------------------------------------------------------------
        nn.Conv2d(32,64,kernel_size = 3, stride=1, padding = 1),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(inplace = True),
        #------------------------------------------------------------------
        nn.Conv2d(64,64,kernel_size = 3, stride=1, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        #------------------------------------------------------------------
        nn.Conv2d(64,128,kernel_size = 3, stride=1, padding = 1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(inplace = True), #output shape is [1,128,1,1]
        #------------------------------------------------------------------
        nn.Flatten(),
        nn.Linear(in_features=128,out_features=128,bias = True),
        nn.ReLU(inplace = True)
    )
  def forward(self,x):
      return self.encode(x)