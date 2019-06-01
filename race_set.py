from hhhh_haha_v3 import *
from data_loader import get_loader
import argparse
from pickletools import optimize
import torchvision
import torchvision.models as models
#from torchvision.models import densenet121

import pickle
import numpy




parser = argparse.ArgumentParser()
 #data/CelebA_nocrop/images                                    scratch/Face\ Images/
parser.add_argument('--celeba_image_dir', type=str, default='/scratch/DataSets/Face\ Images/') #'data/CelebA_nocrop/images'   'data/test_set2'
parser.add_argument('--attr_path', type=str, default='/scratch/chenmao/demographic-others-labels-1.csv')   #'data/list_attr_celeba.txt'        race_label.txt   'data/test_attr.txt'
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the new dataset',
                        default=['Race'])
                        #['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the new dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--batch_size', type=int, default=20, help='mini-batch size')
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
if torch.cuda.is_available():
    d.cuda()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

d.load_state_dict(torch.load('data/models/50150-G.pkl'))

#print("d.state_dict() = " ,d.state_dict())  
#inf.close()
j=1
for epoch in range(1):  # loop over the dataset multiple times
    running_acc = 0.
    
    for i, (img, label) in enumerate(trainloader, 0):
           print("i = ",i)
           img = Variable(img.type(FloatTensor))
    
           label = Variable(label.type(FloatTensor))
    
           "variable is one step which used to transform the geshi"
           print("here")
           output = d(img)
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
           print("output = ", output)
           print("label = ", label)
           correct_num = (output == label).sum()
    
            
           running_acc += correct_num.data
           record = i
    #        if i > 0:
    #           break  

ACC = float(running_acc) / float(20)
print ("batch_size: %d"   %(config.batch_size))
print("[%d/%d]  ACC_num %d Acc: %.5f" %(record, j, running_acc,100*ACC))
