import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.models import densenet121

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
       
        self.conv1 = nn.Sequential(  # input shape (16, 3, 128, 128)   16,3,64,64      new [4 ,3 ,128,128]                     
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1), #output 16*3*64*64     16,3,32,32     new [4 ,3 ,64,64]  
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  #output 16*3*31*31    16,3,15,15        new [4 ,3 ,31,31]  
            nn.BatchNorm2d(3),              
        )

        self.conv2 = nn.Sequential( # input shape (16, 3, 15, 15)   new [4 ,3 ,31,31]  
            nn.Dropout2d(0.5),
            nn.Linear(3 * 31 * 31, 64),  # output shape (32, 14, 14) 
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(64, 64),  # output shape (32, 7, 7)
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(),
        )

    def forward(self, x):

         x = self.conv1(x)
         x = x.view(x.size(0), -1) # 16, 3*31*31  new [4 ,3 ,31,31]
         x = self.conv2(x)
         out = x

         return out