import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.models import densenet121

class CNN_net(nn.Module):
    def __init__(self, d ):
        super(CNN_net, self).__init__()
        #self.densenet_layer = nn.Sequential(*list(d.children())[:-2])
        #torch.nn.functional.linear(input, weight, bias=None)
        self.main = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            
            #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True  (1, 4, 9)
            "picture size 128*128*3*16"
            nn.Conv2d(3, 3, 3, 2, 1),  #Conv2D(3,kernel_size=(3, 3),strides=(2, 2),activation='relu',padding='same',input_shape=input_dim))
            "picture size 64*64*3*16"
            nn.LeakyReLU(0.1, inplace=True),
            #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            nn.MaxPool2d(3,8),#class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)[source]
            "picture size 32*32*3*16"
            #class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)[source]
            nn.BatchNorm2d(3),
          
            nn.Dropout2d(0.5), #torch.nn.functional.dropout2d(input, p=0.5, training=False, inplace=False)[source]
            #class torch.nn.Linear(in_features, out_features, bias=True)
            nn.Linear(32,64),
            nn.Dropout2d(0.5),
            nn.Linear(64,64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64,1),
            nn.LeakyReLU(0.1, inplace=True), 
            nn.Sigmoid(), 
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#                         # [-1, 256, 8, 8]
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, inplace=True),
#             
#             #nn.densenet121(true),#torchvision.models.densenet121(pretrained=False, **kwargs)
#             # [-1, 512, 4, 4]
#             nn.Conv2d(256, 512, 4, 2, 1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1, inplace=True),
# 
#             nn.Conv2d(512, 1024, 4, 2, 1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.1, inplace=True),
#           
#             # [-1, 1 + cc_dim + dc_dim, 1, 1]
#             #nn.Conv2d(1024, 1, 8, 1, 0),
#             nn.MaxPool2d(3,8),#class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)[source]
#             nn.Dropout2d(0.5), #torch.nn.functional.dropout2d(input, p=0.5, training=False, inplace=False)[source]
#             #nn.densenet121(true),#torchvision.models.densenet121(pretrained=False, **kwargs)
#             
#             nn.Conv2d(1024, 1, 1, 1, 0),
        )
        
      
        # run nn.module initial 
        #super(CNN_net, self).__init__()
        
#         
#         self.conv = nn.Conv2d(3, 1, 3, 2, 1)
#         # even pool class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) DOWN sample  stride times
#         self.pool = nn.AvgPool2d(2, 2)
#         # all_connect then softmax   class torch.nn.Linear(in_features, out_features, bias=True)
#         #self.fc = nn.Linear(4*4*1, 1024)
#         self.leaky = nn.LeakyReLU(0.1,inplace = true)
#         "Applies the Softmax function to an n-dimensional input Tensor "
#         "rescaling them so that the elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1"
#         self.softmax = nn.Softmax()
 
    def forward(self, x):
         #out = self.densenet_layer(x)
         
         out = self.main(x)#.squeeze(out,1)
#          out = out.squeeze(1)
#          out = out.squeeze(1)
         #print("out.size = ", out.size())
         "out.size =  torch.Size([16, 1, 1]) with kernel = 8"
         "out.size =  torch.Size([16, 5, 5]) with kernel = 4"
         
         #out = F.softmax(out)
         #print("out_after_sigmoid",out)
#         # 2D converge ->sigmoid stimulate ->pool    
#         out = self.conv(x)
#         
#         out = F.sigmoid(out)
#         out = self.pool(out)
#         
#         #torch.Size([16, 1, 32, 32])
#         print("size = ", out.size())
#         
#         # (batchSize*filterDim*featureDim*featureDim->batchSize*flat_features) change feature dimension 
#         print("check the num_flat_feature",self.num_flat_features(out))   #1024  this will resize to 1024*sth.
#         out = out.view(-1, self.num_flat_features(out))
#         print("out = ", out)
#         print("out_after_view_size = ", out.size())
#         # all_connect  softmax process
#         out = self.leaky(out)
#         print("out_leaky = ", out)
#         print("out_leaky_after_view_size = ", out.size())
#         out = self.softmax(out)
#         print("out_softmax = ", out)
#         print("out_softmax_after_view_size = ", out.size())
#         
         return out
#     def num_flat_features(self, x):
#         # 4D feature,first D is batchSize
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#     