from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import *
from keras.layers import *
from keras.callbacks import  EarlyStopping
import os
import torch 
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.utils.data as utils

X_all, Y_all = loadlocal_mnist('../input/kuzushi/train-images-idx3-ubyte', '../input/kuzushi/train-labels-idx1-ubyte')
X_test, Y_test= loadlocal_mnist("../input/testing/t10k-images-idx3-ubyte", "../input/testing/t10k-labels-idx1-ubyte")
Y_all= np.reshape(Y_all, (-1, 1))
Y_test= np.reshape(Y_test, (-1, 1))
X_all=X_all/255.0
X_test=X_test/255
scaler = StandardScaler(); scaler.fit(X_all)
X_all=scaler.transform(X_all); X_test= scaler.transform(X_test)
X_all= np.reshape(X_all, (X_all.shape[0],28,-1, 1));X_test= np.reshape(X_test, (X_test.shape[0],28,-1, 1)) 
Y_all = to_categorical(Y_all); Y_test= to_categorical(Y_test)

def model2(input_shape):
    X_input = Input(input_shape)
    X= ZeroPadding2D(padding=(2, 2))(X_input)
    
    X= Conv2D(32,kernel_size=(7, 7), strides=(1,1),padding='valid', use_bias=False)(X)
    X= BatchNormalization(axis=3)(X)
    X= MaxPool2D()(X)
    X= Activation('relu')(X)
    
    X= Conv2D(64,kernel_size=(5, 5), strides=(1,1),padding='valid',use_bias=False)(X)
    X= BatchNormalization(axis=3)(X)
    X= Activation('relu')(X)
    
    X= Conv2D(128,kernel_size=(3, 3), strides=(2,2),padding='valid',use_bias=False)(X)
    X= BatchNormalization(axis=3)(X)
    X= MaxPool2D()(X)
    
    X= Activation('relu')(X)
    X= Flatten()(X)
    
    X= Dense(512,activation='relu')(X)
    X= Dropout(rate=0.2)(X)
    X= Dense(10, activation='softmax')(X)
    
    cnn= Model(inputs=X_input, outputs=X)
    return cnn

cnn=model2((28,28,1))
cnn.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(x=X_all, y=Y_all, batch_size=64, epochs=20,verbose=True, validation_data=(X_test,Y_test))
cnn.evaluate(x= X_test,y=Y_test)