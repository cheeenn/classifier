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
from hhhh_haha_v2 import *
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
                        default=['Pale_Skin'])
                        #['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--model_save_dir', type=str, default='data/models') 


config = parser.parse_args()

trainloader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

cuda = True if torch.cuda.is_available() else False

#optimizer = Optimizer(params, defaults);

densenet = models.densenet121(pretrained=True) #torchvision.models.densenet121(pretrained=False, **kwargs)
#model = CNN_net(densenet)
d = CNN_net()
d_opt = optim.Adam(d.parameters(),0.001 , [0.9, 0.999]) 

if torch.cuda.is_available():
    d.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Loss = torch.nn.MSELoss(reduce=True, size_average=False)  # reduce true -- one   false -- a batch 
#transform the loss to the cuda usable
if cuda:
    Loss.cuda()
    
epoches = 30;
#print ("test = " ,98000/float(200704))
for j in range(epoches):
    running_loss = 0.
    running_acc = 0.
    print("trainloader") 
    # enumerate pick one each time
    for i, (img, label) in enumerate(trainloader):
        #print("training = ", i)
       #transform to variable
        # print("img = " ,img)   all pixel of this image , vector[ 0.7333,  0.7412,  0.7647,  ..., -0.0980,  0.4431,  0.4196]
        #img = Variable(img)
        #print("training", i) 
        img = Variable(img.type(FloatTensor))
        #print("img_VAR = " ,img)
        # batchsize channel width height
        #print("size zero= " ,img.size(0))
        #print("size one = " ,img.size(1))
        #print("size three= " ,img.size(2))
       # print("size four = " ,img.size(3))
        #print("label = " ,label)  [1., 0., 0., 0., 1.],  current we choose 5 feature in date_loader
        label = Variable(label.type(FloatTensor))
        #print("label_var = " ,label) 
       # print("label.size = ", label.size())
        "variable is one step which used to transform the geshi"
        
        #class torch.optim.Optimizer(params, defaults)
       
        
        # feedforward
        output = d(img)
       # print("output_inclassifier = ", output)
       # print("out.classi.size = ", output.size())
        "resize"
       # output = output.view(output.shape[0],-1) 
        # loss function is MSE 
        loss = Loss(output, label)  #loss =  tensor(11.5511, device='cuda:0', grad_fn=<MseLossBackward>)
        #print ("label " , label)
        
        #print("loss = ", loss)
       
        "Clears the gradients of all optimized torch.Tensor s.  just in case to find a local optimise"
        d_opt.zero_grad()
                
        # backward to adjust the loss 
        loss.backward()
        "Performs a single optimization step."
        d_opt.step()
        #print("output_inclassifier_afterallprocess = ", output)
        
        "record current loss"  # record current lost and batchSize data
        running_loss += loss.data
        #print("loss.data[0] = ", loss.data)
        #print("loss.size = ", loss.size())
        #print("runing_loss = ", running_loss)   
        #_, predict = torch.max(output, 1)    # pick max of output according row, if second var is 0, pick max according column
        output = torch.sign(output)
        output = (output+1)/2
        #print("loss_after_sign_to_one_zero = ", loss) #loss_after_sign_to_one_zero =  tensor(1., device='cuda:0', grad_fn=<DivBackward0>)
        #print("output ", output)
        #predict = loss
        #print("label = ", label)
        correct_num = (output == label).sum()
        #print("correct_num = ", correct_num)   #correct_num =  tensor(0, device='cuda:0')
        
        running_acc += correct_num.data
        record = i;
        #if i > 10:
        #  break  
 
    #print("training", record) 
    # calculate and precise percentage
    #running_loss /= ( record)
    "(record*config.batch_size)"
    ACC = float(running_acc) / float(200704)
    #soft max
 # set the output pale skin is one, and not pale skin is zero
 
   
    print ("batch_size: %d"   %(config.batch_size))
    print("[%d/%d/%d] Loss: %.5f ACC_num %d Acc: %.5f" %(i+1, j, epoches, running_loss, running_acc,100*ACC))
    
"save the model"    
#g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
#d_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(i+1))
d_path = os.path.join(config.model_save_dir, '{}-G.pkl'.format(i+1))
torch.save(d.state_dict(), d_path)