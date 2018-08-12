# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:09:55 2017

@author: WTG
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import concatenate
from pandas import DataFrame
from pandas import read_csv
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import random
random.seed(10)

train=read_csv('C:/Users/WTG/train1.csv')
ntrain = int(len(train))
train=train[:ntrain]
#train[np.isnan(train)]=0
a=train.columns.values

test_initial=read_csv('C:/Users/WTG/test1.csv')
ntest = int(len(test_initial))
test_initial=test_initial[:ntest]
test=test_initial.drop('order_id', 1)
test=test.drop('product_id', 1)
#test[np.isnan(test)]=0

scaler = MinMaxScaler(feature_range=(0, 1))
train=scaler.fit_transform(train)
test=scaler.fit_transform(test)
train=DataFrame(train)
train.columns=a
test=DataFrame(test)

def train_transform(data,SEQ_LENGTH = 16):
    num_attr = data.shape[1]
    result = np.empty((len(data) - SEQ_LENGTH - 1, SEQ_LENGTH, num_attr))
    y = np.empty(len(data) - SEQ_LENGTH - 1)

    for index in range(len(data) - SEQ_LENGTH - 1):
        result[index, :, :] = data[index: index + SEQ_LENGTH]
        y[index] = data.iloc[index + SEQ_LENGTH + 1].reordered

    xtrain = result[:len(data), :, :]
    ytrain = y[:len(data)]
    return xtrain, ytrain

xtrain, ytrain=train_transform(train)
xtrain=xtrain[:,:,0:19]

def test_transform(data,SEQ_LENGTH = 16):
    num_attr = data.shape[1]
    result = np.empty((len(data) - SEQ_LENGTH - 1, SEQ_LENGTH, num_attr))
    
    for index in range(len(data) - SEQ_LENGTH - 1):
        result[index, :, :] = data[index: index + SEQ_LENGTH]

    xtest = result[:len(data), :, :]
    return xtest

xtest=test_transform(test)



def train_model(xtrain,ytrain,SEQ_LENGTH=16,N_HIDDEN_1=256,N_HIDDEN_2=64):
    # SEQ_LENGTH = 16  # Sequence Length,
    # N_HIDDEN = 512  # Number of units in the hidden (LSTM) layers
    # num_attr = 22  # Number of predictors used for each trading day
    num_attr = xtrain.shape[2]
    model = Sequential()
    model.add(LSTM(N_HIDDEN_1, return_sequences=True, activation='tanh', input_shape=(SEQ_LENGTH, num_attr)))
    model.add(Dropout(0.2))
    model.add(LSTM(N_HIDDEN_2, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, batch_size=128, epochs=5, validation_split=0.1)
    return model

model = train_model(xtrain,ytrain)
model.summary()
predicted=model.predict(xtest)
predicted=DataFrame(predicted)
predicted.columns = ['reordered']
test_initial=pd.concat([test_initial,predicted], axis=1)

sub=DataFrame()
sub.loc[:, 'prediction']=predicted
sub.loc[:, 'product_id']=test_initial.loc[:, 'product_id']
sub.loc[:, 'order_id']=test_initial.loc[:, 'order_id']

sub.to_csv("sub2.csv", index=False)
