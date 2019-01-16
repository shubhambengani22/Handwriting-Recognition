#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:37:25 2018

@author: shubham
"""

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')

# All pixel values - all rows and column 1 (pixel0) to column 785 (pixel 783)
x_train = (train.iloc[:,1:].values).astype('float32') 
# Take a look at x_train
x_train

# Labels - all rows and column 0
y_train = (train.iloc[:,0].values).astype('int32') 

# Take a look at y_train
y_train

test = pd.read_csv('test.csv')

x_test = test.values.astype('float32')

from keras.utils import to_categorical

num_classes = 10

# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
np.reshape
# Reshape input data from (samples, 28, 28) to (samples, 28*28)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes)

# Take a look at the dataset shape after conversion with keras.utils.to_categorical
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(rate=0.3))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=64)

pred = model.predict(x_test)
s = pd.read_csv('sample_submission.csv')
#from sklearn.metrics import confusion_matrix, classification_report
#print(confusion_matrix(s.Label, a))
a = []
for i in range(len(pred)):
    a.append(pred[i].argmax())
s.Label = a
s.to_csv('Submission4.csv', index = False)


a = pd.DataFrame(a)
a.to_csv('Submission.csv')
a = np.array(a)
b = np.array(b)
b = np.reshape(b, (28000, 1))




c = pd.DataFrame(c,columns = ['Label'])














from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
