from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pandas as pd
import imghdr
from _ast import Continue


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        d = pd.read_csv(self.attr_path, index_col=False)
        print("csv dim is = ",d.shape)
        #for line in d.values:
           # print(line)
        #lines = [line.rstrip() for line in open(self.attr_path, 'r')]  this is for txt 
        lines =[line for line in d.values]
       # print("1 =",lines[0])  this is work for read the csv, 0 is the csv file second line
        #print("2 =",lines[1])
       # print("3 =",lines[2])
        all_attr_names = lines[0]
        print("all attribute name = ", all_attr_names)
        #all_attr_names = lines[1].split()    this is for txt
        ne =0;
    
        for i, attr_name in enumerate(all_attr_names):
            print( "attribute name = %s , i = %s " %(attr_name ,i)  )
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            #print("lines = " , line) #this is the each line of the text
            filename = lines[i][0]
            #print ("filename = ",filename) this is work , for each name is correct
            #if len(split) < 4 : continue
            
            #print("split = ",split)   split =  ['Google_1_Steve', 'Westbrooks_3_oval.jpg', '804', '0', '1', '4', '3']
            #if split[1].isdigit() == False:
            #    filename = split[0]+' '+split[1]
            #filename = split[0]
            #values = split[1:]
            #else :
           #     filename = split[0] 
            #    values = split[0:]
            #print("value = ", values)    
            #print( "file name = ", filename)
            label = []
            
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                value = lines[i][idx]
                #print('idx = ', idx ) for race is 5
                label.append(value)  # label.append(values[idx] == '1')
                
            #print("label = " , label ) label is correct here, one attr so just one label each time
            
            ###########################################      
           # each_path = self.image_dir + filename  this doesn't work because the path is not correct i think 
          #  print(each_path)
          #  if  os.path.exists(each_path):  
           #     print('exist')
           # else: 
           #     print('not exit this image')
           #     ne = ne+1


            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
                self.train_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

            #print (self.train_dataset) store work properly 
        #print('number of missing image = ' , ne)

        print('Finished preprocessing the new dataset...')
        

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
   
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        print("number of images = ", self.num_images)
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))                        #center crop
    transform.append(T.Resize(image_size))                           #resize
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
    print("image dir is ",image_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader