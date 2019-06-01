from hhhh_haha_v3 import *
from data_loader import get_loader
import argparse
from pickletools import optimize
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import os
#from torchvision.models import densenet121
from torch import optim
import pickle
import numpy




parser = argparse.ArgumentParser()
 #data/CelebA_nocrop/images                                    scratch/Face\ Images/
parser.add_argument('--celeba_image_dir', type=str, default='/home/research/chenmao/GitHub/B_classifier/DataSets/FaceImages') #'data/CelebA_nocrop/images'   'data/test_set2'
parser.add_argument('--attr_path', type=str, default='/home/research/chenmao/GitHub/B_classifier/real_race_label_short.txt')   #'data/list_attr_celeba.txt'        race_label.txt   'data/test_attr.txt'
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the new dataset',
                        default=['Race'])
                        #['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the new dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--model_save_dir', type=str, default='data/models') 


config = parser.parse_args()

trainloader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

#model = open ('data/models/50150-G.pkl','rb')
#inf=pickle.load(model)
#inf=torch.load('data/models/50150-G.pkl')
cuda = True if torch.cuda.is_available() else False
d = CNN_net()
d_opt = optim.Adam(d.parameters(),0.001 , [0.9, 0.999], )

if torch.cuda.is_available():
    d.cuda()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Loss = torch.nn.MSELoss(reduce=True, size_average=False)  # reduce true -- one   false -- a batch 
#d.load_state_dict(torch.load('data/models/50150-G.pkl'))
if cuda:
    Loss.cuda()
    
#print("d.state_dict() = " ,d.state_dict())  
#inf.close()
epoches=30
for epoch in range(epoches):  # loop over the dataset multiple times
    running_loss = 0.
    running_acc = 0.
    for i, (img, label) in enumerate(trainloader, 0):
           #print("i = ",i)
           #print("in loop image size = " , img.size)
           #print("in loop image shape =", img.shape)   useful for input shape
           img   = Variable(img.type(FloatTensor))
           
           label = Variable(label.type(FloatTensor))
           #print("label = ", label)
           "variable is one step which used to transform the geshi"
           #print("here")
           output = d(img)
           
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
           o = 0;
           for out in output:     
             if    out > 0.84: output[o] = 6 
             elif  out > 0.7:  output[o] = 5 
             elif  out > 0.56: output[o] = 4 
             elif  out > 0.42: output[o] = 3
             elif  out > 0.28: output[o] = 2
             elif  out > 0.14: output[o] = 1 
             else: output[o] = 0
             o=o+1
           o=0; 
           #print("output = ", output)
           #print("label = ", label)
           correct_num = (output == label).sum()
    
            
           running_acc += correct_num.data
           record = i
    #        if i > 0:
    #           break  

ACC = float(running_acc) / float(1761)
print ("batch_size: %d"   %(config.batch_size))
print("[%d/%d/%d] Loss: %.5f ACC_num %d Acc: %.5f" %(i+1,epoch, epoches, running_loss, running_acc,100*ACC))



d_path = os.path.join(config.model_save_dir, '{}-Race.pkl'.format(i+1))
torch.save(d.state_dict(), d_path)