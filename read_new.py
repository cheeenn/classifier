import imghdr
import os

label =[]
label.append(1)
label.append(2)

for x in range(6):
    if x==5:
        continue
 
    print x

print label.__len__()
filename = 'Google_1_Priscilla Terry_3_oval.jpg'
dir = '/scratch/DataSets/Face\ Images/' +filename
'C:/Users/32961/Desktop/GAN/StarGAN-master/classifier/DataSets/FaceImages/'
'/scratch/DataSets/Face\ Images/'
#print os.path.dirname('Adam_Ng_13_oval.jpg')
if  os.path.exists(dir):  print('exist')
else :  print('not exist')

#if imghdr.what('race_label.txt'): print("it is jpg")
#else: print("none")

#if os.access("C:\Users\32961\Desktop\GAN\StarGAN-master\classifier\DataSets\Face Images/Adam_Ng_13_oval.jpg", os.F_OK):
   # print "Given file path is exist."
 

