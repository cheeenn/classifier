import os
'C:/Users/32961/Desktop/GAN/StarGAN-master/DataSets/Face Images/'#100000
'C:/Users/32961/Desktop/GAN/StarGAN-master/classifier/DataSets/FaceImages/' #601
path ='C:/Users/32961/Desktop/GAN/StarGAN-master/DataSets/Face Images/'#100000

filelist = os.listdir(path)
total_num=len(filelist)

print ("image number = " ,total_num)

i=0
str = "this is string example....wow!!! this is really string"
print str.replace(" ", "")
print str.replace("is", "was", 3)

for item in filelist:
    ori_name = os.path.join(os.path.abspath(path),item)
    #item.replace(' ' ,'')
    #print(item)
    ori_new  = os.path.join(os.path.abspath(path),item.replace(' ' ,''))
    os.rename(ori_name,ori_new)
    print ori_new
    i = i+1
print("conver %d jpg to no space version" %(i))