import os
import torch
import torchvision
import numpy as np

from torch.autograd import Variable
from torch import optim
#from torch.optim import Optimizer 

import torch
import torch.nn as nn
import torch.nn.functional as F
from hhhh_haha_v3 import *
from data_loader import get_loader
import argparse
from pickletools import optimize
import torchvision
import torchvision.models as models
from torchvision.models import densenet121

parser = argparse.ArgumentParser()
 #data/CelebA_nocrop/images
parser.add_argument('--celeba_image_dir', type=str, default='data/CelebA_nocrop/images')
parser.add_argument('--attr_path', type=str, default='data/list_attr_celeba.txt')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Male']) #Pale_Skin Male
                        #['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--image_size', type=int, default=64, help='image resolution')
parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--model_save_dir', type=str, default='data/models') 


config = parser.parse_args()

trainloader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

cuda = True if torch.cuda.is_available() else False

d = CNN_net()
d_opt = optim.Adam(d.parameters(),0.001 , [0.9, 0.999], )

if torch.cuda.is_available():
    d.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Loss = torch.nn.MSELoss(reduce=True, size_average=False)  # reduce true -- one   false -- a batch 

if cuda:
    Loss.cuda()
    
epoches = 8;
print ("this is for  = " ,config.selected_attrs)
for j in range(epoches):
    running_loss = 0.
    running_acc = 0.
    print("trainloader")
    for i, (img, label) in enumerate(trainloader):

        img = Variable(img.type(FloatTensor))

        label = Variable(label.type(FloatTensor))

        "variable is one step which used to transform the geshi"

        output = d(img)

        "resize"
        #print("d.state_dict()",d.state_dict())
#           [[-0.0264, -0.0576,  0.0558],
#           [ 0.1457, -0.1097,  0.1192],
#           [-0.1602,  0.0859, -0.1198]]]], device='cuda:0')), 
#           ('conv1.0.bias', tensor([ 0.0977, -0.1315, -0.0772], device='cuda:0')), 
#           ('conv1.3.weight', tensor([0.0348, 0.6806, 0.8784], device='cuda:0')),
#           ('conv1.3.bias', tensor([0., 0., 0.], device='cuda:0')), 
#           ('conv1.3.running_mean', tensor([0.0066, 0.0082, 0.0099], device='cuda:0')),
#           ('conv1.3.running_var', tensor([0.9006, 0.9042, 0.9014], device='cuda:0')), 
#           ('conv1.3.num_batches_tracked', tensor(1, device='cuda:0')), 
#           ('conv2.1.weight', tensor([[ 0.0036,  0.0111, -0.0040,  ..., -0.0054,  0.0099, -0.0006],
        # output = output.view(output.shape[0],-1)
 
        # loss function is MSE 
        loss = Loss(output, label)  #loss =  tensor(11.5511, device='cuda:0', grad_fn=<MseLossBackward>)
       
        "Clears the gradients of all optimized torch.Tensor s.  just in case to find a local optimise"
        d_opt.zero_grad()
                
        # backward to adjust the loss 
        loss.backward()
        "Performs a single optimization step."
        d_opt.step()
        #print("output_inclassifier_afterallprocess = ", output)

        "record current loss"  # record current lost and batchSize data
        running_loss += loss.data
#         output = torch.sign(output)
#         output = (output + 1) / 2
        #print("output = ",output)
        #print("output1 = ",output[1])
        "output here output =  tensor([[0.5042],[0.4662],[0.4619], [0.4599]], device='cuda:0', grad_fn=<SigmoidBackward>) "
        o = 0;
        for out in output:     
             if out > 0.5: output[o] = 1 
             else: output[o] = 0
             o=o+1
        o=0; 
        #print("output = ",output)  is good to go 0,1
        correct_num = (output == label).sum()
        #print("correct_num = ", correct_num)   #correct_num =  tensor(0, device='cuda:0')
        
        running_acc += correct_num.data
        record = i;
#         if i > 3000:
#           break  
 
    #print("training", record) 
    # calculate and precise percentage
    #running_loss /= ( record)
    "(record*config.batch_size)"
    #ACC = float(running_acc) / float(3000*4)
    ACC = float(running_acc) / float(200704)
    #soft max
 # set the output pale skin is one, and not pale skin is zero
 
   
    print ("batch_size: %d"   %(config.batch_size))
    print("[%d/%d/%d] Loss: %.5f ACC_num %d Acc: %.5f" %(i+1, j, epoches, running_loss, running_acc,100*ACC))
    
"save the model"    
#g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
#d_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(i+1))
d_path = os.path.join(config.model_save_dir, '{}-Gender.pkl'.format(i+1))
torch.save(d.state_dict(), d_path)