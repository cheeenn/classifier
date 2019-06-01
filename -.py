import pandas as pd
from os import listdir
from os.path import isfile, join

labels = 'demographic-others-labels-1.csv'
images = '/Users/fferdinando3/Repos/GAN/datasets/Faces/data/images'

onlyfiles = [f for f in listdir(images) if isfile(join(images, f))]
d = pd.read_csv(labels, index_col=False)
imgs_with_labels = d[['Filename']].values.flatten()

for img_lab in imgs_with_labels:
    print(img_lab.split('_'))
    found = False
    for file in onlyfiles:
        if all(s in file for s in img_lab.split('_')):
            print(img_lab, ' --> ', file)
            found = True
    assert found