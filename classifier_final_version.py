
from hhhh_haha_v3 import *
from data_loader import get_loader

#3546
########################################import data#########################################
#import sys
#sys.path.append('/Users/fferdinando3/Repos/GAN/bias-rm-gan/src')
#from data_loader import get_loader
attrpath='data/list_attr_celeba.txt'
imgpath='data/CelebA_nocrop/images'
trainloader = get_loader(imgpath, attrpath, ['Pale_Skin'], 128, 128, 4, 'CelebA', 'train', 2)

classes = ('Pale_Skin', 'Non-Pale_Skin')

print(len(trainloader))

import matplotlib.pyplot as plt
import numpy as np
import torchvision
########################################define image########################################
# functions to show an image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def imshow(img):
    img = denorm(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images,pad_value=30))
# print labels
print('\t' + '   '.join('%5s  ' % classes[int(labels[j])] for j in range(4)))
################################# Define a Loss function and optimizer######################

net = CNN_net()

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SDG(net.parameters(), lr=0.001, momentum=0.9)


from torch.autograd import Variable
import torch

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
        #if i > 10000: break

print('Finished Training')
############################## Test the network on the data#####################
'Let us print some images from the test set'
testloader = get_loader(imgpath, attrpath, ['Pale_Skin'], 128, 128, 4, 'CelebA', 'test', 2)

dataiter = iter(testloader)
images, labels = dataiter.next()
labels = np.round(labels.flatten().detach().numpy()).astype(int)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
predicted = np.round(outputs.flatten().detach().numpy()).astype(int)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

###############################  Check accuracy ############################### 
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        predicted = np.round(outputs.flatten().detach().numpy()).astype(int)
        total += labels.size(0)
        correct += (labels.flatten().detach().numpy() == predicted).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))






