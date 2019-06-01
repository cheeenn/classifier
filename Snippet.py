
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from Callbacks import Callbacks
from Config import Config

# get config
config = Config().config
# Model checkpoint
config.period = 20
# ReduceLROnPlateau
config.lr_min_lr = 1e-07
config.lr_epsilon = 0.001
config.lr_factor = 0.5
config.lr_patience = 3
# Early stopping
config.es_min_delta = 0.001
# Training and testing
config.epochs = 40
config.batch_size = 4
num_images = 20000
batch_size_mult = 2
# Optimizer
optimizer = Adam(lr=0.001,  # 0.001
                 beta_1=0.9,  # 0.9
                 beta_2=0.999,  # 0.999
                 epsilon=1e-08,  # 1e-08
                 decay=0.0)  # 0.0

# For the case in which we change the batch size:
config.change_lr = False
config.change_bs = True
config.es_patience = 3

# get directories
log_dir = '/Users/fferdinando3/Repos/GAN/celeba-classifier/log/'
data_dir = '/Users/fferdinando3/Repos/GAN/datasets/'
image_dir = '/Users/fferdinando3/Repos/GAN/datasets/celebA/imgs/'
annotations = '/Users/fferdinando3/Repos/GAN/datasets/celebA/annotations/list_attr_celeba.txt'

# todo: Load data and pre-process annotations into dictionary. 
filename_gender_dict = {}

import csv
import pandas as pd

d = pd.read_csv(annotations, delim_whitespace=True)
d_clear = d#[(d.Blurry != -1)]
d_ann = d_clear[['name', 'Pale_Skin']]
d_ann.loc[d_ann['Pale_Skin'] == -1, 'Pale_Skin'] = 0
#d_ann

data_dict = dict()
for index, row in d_ann.iterrows():
   data_dict[row[0]] = row[1]
x_data = []
y_data = []
keys = list(d_ann.name)
i = 1
images = os.listdir(image_dir)
shuffle(images)

# load x and y data
try:
    for file in images:
        if i > num_images:
            break
        if file in keys:
            filepath = os.path.join(image_dir, file)
            im_arr = mpimg.imread(filepath)
            if im_arr.shape == (218, 178, 3):
                x_data.append(im_arr)
                y_data.append(np.array(data_dict[file]))
                i += 1
except Exception as e:
    print(str(e))
	
x_data = np.array(x_data)
y_data = np.array(y_data)
# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
num_train = int(x_train.shape[0] * 0.8)


# build model
input_dim = x_train[0].shape

model = Sequential()
model.add(Conv2D(3,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 activation='relu',
                 padding='same',
                 input_shape=input_dim))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics = ['accuracy'])
model.summary()


# training

# get callbacks
callbacks = Callbacks(config, log_dir).callbacks
print('callbacks:')
for callback in callbacks:
    print('\t', callback)

# set variables
val_loss = []
val_acc = []
loss = []
acc = []
lr = []
bs = []
epoch_iter = 1
max_epochs = config.epochs
batch_size = config.batch_size

# train model
if config.change_lr:  # reduce_lr callback takes care of everything for us
    print('Will change learning rate during training, but not batch size')
    print('Training model...')
    history = model.fit(x_train,
                        y_train,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks)
    # store history (bs is constant)
    val_loss += history.history['val_loss']
    val_acc += history.history['val_acc']
    loss += history.history['loss']
    acc += history.history['acc']
    lr += history.history['lr']
    bs = [batch_size for i in range(len(history.epoch))]

elif config.change_bs:  # need to manually stop and restart training
    print('Will change batch size during training, but not learning rate')
    while max_epochs >= epoch_iter:
        print(f'Currently at epoch {epoch_iter} of {max_epochs}, batch size is {batch_size}')
        epochs = max_epochs - epoch_iter + 1
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=1,
                            callbacks=callbacks)
        # store history
        val_loss += history.history['val_loss']
        val_acc += history.history['val_acc']
        loss += history.history['loss']
        acc += history.history['acc']
        bs += [batch_size for i in range(len(history.epoch))]

        # update training parameters
        epoch_iter += len(history.epoch)
        batch_size *= batch_size_mult
        batch_size = batch_size if batch_size < num_train else num_train

    # store lr history as constant (because it is)
    lr = [0.001 for i in range(len(bs))]

else:
    print('Will not change learning rate nor batch size during training')
    print('Training model...')
    history = model.fit(x_train,
                        y_train,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks)
    # store history (bs is constant)
    val_loss += history.history['val_loss']
    val_acc += history.history['val_acc']
    loss += history.history['loss']
    acc += history.history['acc']
    lr = [0.001 for i in range(len(history.epoch))]
    bs = [batch_size for i in range(len(history.epoch))]

print('Completed training')