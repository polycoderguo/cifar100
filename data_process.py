from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
import warnings

from keras.datasets import cifar100
from keras.utils import np_utils, to_categorical

'''
get train and test data from cifar dataset
target_0 and target_1: classes chosen by user
train_1 and train_0 : the rest labels from the two super classes
if deep_learning is set to True: y is transformed to categorical 

An example input :

target_0 = ['bridge', 'castle'] 
target_1 = ['cloud', 'forest']
train_0 = ['house', 'road', 'skyscraper']
train_1 = [ 'mountain', 'plain','sea']
x_train, x_test, y_train, y_test = PreprocessDataset(target_0, target_1, train_0, train_1, False)
'''



def PreprocessDataset(target_0, target_1, train_0, train_1, deep_learning):
    (x_train_org, y_train_org), (x_test_org, y_test_org) = cifar100.load_data(label_mode='fine')
    x_train_org = x_train_org.astype('float32')
    x_test_org = x_test_org.astype('float32')
    # Normalize value to [0, 1]
    #x_train_org /= 255
    #x_test_org /= 255
    # Transform lables to one-hot
    y_train_org = np_utils.to_categorical(y_train_org, 100)
    y_test_org = np_utils.to_categorical(y_test_org, 100)
    # Reshape: here x_train is re-shaped to [channel] × [width] × [height]
    # In other environment, the orders could be different; e.g., [height] × [width] × [channel].
    x_train_org = x_train_org.reshape(x_train_org.shape[0], 32, 32, 3)
    x_test_org = x_test_org.reshape(x_test_org.shape[0], 32, 32, 3)
    
    CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm']
    #Label chose:
    '''target_0, target_1, train_0, train_1'''
    label_target_loc_0 = []
    label_target_loc_1 = []
    label_train_loc_0 = []
    label_train_loc_1 = []
    for item in CIFAR100_LABELS_LIST:
        if item in target_0:
            label_target_loc_0.append(CIFAR100_LABELS_LIST.index(item))
        elif item in target_1:
            label_target_loc_1.append(CIFAR100_LABELS_LIST.index(item))
        elif item in train_0:
            label_train_loc_0.append(CIFAR100_LABELS_LIST.index(item))
        elif item in train_1:
            label_train_loc_1.append(CIFAR100_LABELS_LIST.index(item))
        
    x = np.concatenate((x_train_org,x_test_org), axis = 0)
    y = np.concatenate((y_train_org,y_test_org), axis = 0)
    #set up subclass
    test_0 = []
    test_1 = []
    train_0 = []
    train_1 = []
    for i, item in enumerate(y):
        if np.argmax(item) in label_target_loc_0:
            test_0.append(x[i])
        elif np.argmax(item) in label_target_loc_1:
            test_1.append(x[i])
        elif np.argmax(item) in label_train_loc_0:
            train_0.append(x[i])
        elif np.argmax(item) in label_train_loc_1:
            train_1.append(x[i])
    
    x_train = np.concatenate((train_0, train_1), axis = 0)
    #print(x_train.shape)
    #x_train = x_train.reshape((x_train.shape[0])*600,32,32,3)
    x_test = np.concatenate((test_0, test_1), axis = 0)
    #x_test = x_test.reshape((test_0.shape[0]+test_1.shape)*600, 32*32*3)
    #print(len(test_0))
    y_train = np.concatenate((np.zeros(len(train_0)), np.ones(len(train_1))), axis = 0)
    y_test = np.concatenate((np.zeros(len(test_0)), np.ones(len(test_1))), axis = 0)
    if deep_learning: 
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)
    return x_train, x_test, y_train, y_test
