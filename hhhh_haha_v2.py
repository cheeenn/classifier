import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.models import densenet121

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (16, 3, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1,), #output 16*3*64*64
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  #output 16*3*32*32   3*32*32 =3072
            nn.BatchNorm2d(3),
        )

        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Dropout2d(0.5),
            nn.Linear(3 * 31 * 31, 64),  # output shape (32, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),  # activation
            nn.Linear(64, 64),  # output shape (32, 7, 7)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

         x = self.conv1(x)
         #x size torch.Size([16, 3, 31, 31])
         
         #print ("x size",  x.size())
         x = x.view(x.size(0), -1)#flatten change to one dimension
         
         #x size torch.Size([16, 2883])
         #print ("x size",  x.size())
         x = self.conv2(x)
         out = x

         return out