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
from hhhh_haha import *
from data_loader import get_loader
import argparse
from pickletools import optimize


class run_classifier():
    def__init__(celeba_image_dir,attr_path,selected_attrs, celeba_crop_size,image_size,batch_size,mode,num_workers):
        self.image_dir = celeba_image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.celeba_crop_size = celeba_crop_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode 