#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
# Mohit Kumar Ahuja                                                             #
# PhD Student                                                                   #
# Simula Research Laboratory                                                    #
# mohit@simula.no                                                               #
# File: All the functions for Regression Testing                                #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

from __future__ import print_function
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras import backend as K
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
import matplotlib.image as mpimg
from scipy import misc
import requests
import scipy.io as scio
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from PIL import *
from scipy import ndimage
from tqdm import tqdm, trange
import os
import tarfile
import gzip
import pickle
from matplotlib.pyplot import imshow
import scipy
from keras import regularizers
import itertools
import tensorflow as tf
import pandas as pd
tf.reset_default_graph()
from keras.utils import np_utils
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
plt.style.use('ggplot')
import datetime
now = datetime.datetime.now
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1)

flags = tf.app.flags
FLAGS = flags.FLAGS

def reg_cnn_model(y_train):
    # create model
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(784), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01) ))
    model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def Mohit_model():
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model

def baseline_model(Train_X, y_train):
    # create model
    num_pixels = Train_X.shape[0]
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def maybe_download(url, dest_dir):
    """Checks if file exists and tries to download it if not"""
    if not os.path.exists(dest_dir):
        print("Creating directory: {}".format(dest_dir))
        os.makedirs(dest_dir)
    filename = url.split("/")[-1]
    dest_filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(dest_filepath):
        print("Downloading", url)
        response = requests.get(url)
        with open(dest_filepath, 'wb') as fil:
            fil.write(response.content)
    return dest_filepath

def load_svhn(data_dir):

    url = "http://ufldl.stanford.edu/housenumbers/"
    print("Loading svhn data from {}".format(data_dir))

    _ = maybe_download(url+"train_32x32.mat", data_dir)
    train_data = scio.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    rows, cols, channels, num_train = train_data['X'].shape
    X_train = train_data['X'].reshape((rows*cols*channels, num_train))
    Y_train = train_data['y'].squeeze()

    _ = maybe_download(url+"test_32x32.mat", data_dir)
    test_data = scio.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    rows, cols, channels, num_test = test_data['X'].shape
    X_test = test_data['X'].reshape((rows*cols*channels, num_test))
    Y_test = test_data['y'].squeeze()

    # Number 0 was initially labeled as 10
    Y_train[Y_train == 10] = 0
    Y_test[Y_test == 10] = 0

    # Scale data to [0.0, 1.0]. Other data standardisation can be done here.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def train_model_mnist(model, train, val, test, num_classes, input_shape, batch_size, epochs, lr):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    Dev_X = val[0].reshape((val[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Dev_X Shape:', Dev_X.shape)
    print(x_train.shape[0], 'train samples')
    print(Dev_X.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    optimizer = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizer,
                  metrics=['accuracy'])

    t = now()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y))
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history_MNIST_REGRESSION_TRAINING_server.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
    print('Training time: %s' % (now() - t))
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def train_model_fmnist(model, train, val, test, num_classes, input_shape, batch_size, epochs, lr):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    Dev_X = val[0].reshape((val[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #print('x_train shape:', x_train.shape)
    #print('Dev_X Shape:', Dev_X.shape)
    #print(x_train.shape[0], 'train samples')
    #print(Dev_X.shape[0], 'validation samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    optimizer = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizer,
                  metrics=['accuracy'])

    t = now()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y))
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history_F_MNIST_REGRESSION_TRAINING_server.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
    print('Training time: %s' % (now() - t))
    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    

def train_model_cifar(model, train, val, test, num_classes, input_shape, batch_size, epochs):
    x_train = train[0]
    Dev_X = val[0]
    x_test = test[0]
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Dev_X Shape:', Dev_X.shape)
    print(x_train.shape[0], 'train samples')
    print(Dev_X.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y))
    print('Training time: %s' % (now() - t))
    start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Total Time for testing = " + str(time.time() - start) + "sec")
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    
def train_model_svhn(model, train, val, test, num_classes, input_shape, batch_size, epochs):
    x_train = train[0]
    Dev_X = val[0]
    x_test = test[0]
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Dev_X Shape:', Dev_X.shape)
    print(x_train.shape[0], 'train samples')
    print(Dev_X.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y))
    print('Training time: %s' % (now() - t))
    start = time.time()
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Total Time for testing = " + str(time.time() - start) + "sec")
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    
    return encoded


        
        
def shuffle_Data(X, Y):

        
    # Shuffling all the Images
    m = X.shape[0]      # number of training examples

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    return shuffled_X, shuffled_Y


def shuffle_Data_2D(X, Y):

    #print('shape before shuffling:', X.shape)
    # Shuffling all the Images
    m = X.shape[1]      # number of training examples
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[permutation]
    #print('shape after shuffling:', shuffled_X.shape)

    return shuffled_X, shuffled_Y

def Print_shape(Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, text):

    print("Shape of " + str(text) + " 0 = " + str(Class0.shape))
    print("Shape of " + str(text) + " 1 = " + str(Class1.shape))
    print("Shape of " + str(text) + " 2 = " + str(Class2.shape))
    print("Shape of " + str(text) + " 3 = " + str(Class3.shape))
    print("Shape of " + str(text) + " 4 = " + str(Class4.shape))
    print("Shape of " + str(text) + " 5 = " + str(Class5.shape))
    print("Shape of " + str(text) + " 6 = " + str(Class6.shape))
    print("Shape of " + str(text) + " 7 = " + str(Class7.shape))
    print("Shape of " + str(text) + " 8 = " + str(Class8.shape))
    print("Shape of " + str(text) + " 9 = " + str(Class9.shape))
 

def Class_Distribution(X, Y, size_of_set, devel_size):
    
    # Classifying thwe data into classwise
    toto = np.where(Y == 0)
    Class0 = X[:, toto]
    toto = np.where(Y == 1)
    Class1 = X[:, toto]
    toto = np.where(Y == 2)
    Class2 =X[:, toto]
    toto = np.where(Y == 3)
    Class3 =X[:, toto]
    toto = np.where(Y == 4)
    Class4 =X[:, toto]
    toto = np.where(Y == 5)
    Class5 =X[:, toto]
    toto = np.where(Y == 6)
    Class6 =X[:, toto]
    toto = np.where(Y == 7)
    Class7 =X[:, toto]
    toto = np.where(Y == 8)
    Class8 =X[:, toto]
    toto = np.where(Y == 9)
    Class9 =X[:, toto]

    # Removing the third Dimension
    Class0 = Class0[:,0,:]
    Class1 = Class1[:,0,:]
    Class2 = Class2[:,0,:]
    Class3 = Class3[:,0,:]
    Class4 = Class4[:,0,:]
    Class5 = Class5[:,0,:]
    Class6 = Class6[:,0,:]
    Class7 = Class7[:,0,:]
    Class8 = Class8[:,0,:]
    Class9 = Class9[:,0,:]
    #Print_shape(Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, text = "Main Class")


    # Dividing the dataset 
    Class0_new = Class0[:,0:size_of_set]
    Class1_new = Class1[:,0:size_of_set]
    Class2_new = Class2[:,0:size_of_set]
    Class3_new = Class3[:,0:size_of_set]
    Class4_new = Class4[:,0:size_of_set]
    Class5_new = Class5[:,0:size_of_set]
    Class6_new = Class6[:,0:size_of_set]
    Class7_new = Class7[:,0:size_of_set]
    Class8_new = Class8[:,0:size_of_set]
    Class9_new = Class9[:,0:size_of_set]
    #Print_shape(Class0_new, Class1_new, Class2_new, Class3_new, Class4_new, Class5_new, Class6_new, Class7_new, Class8_new, Class9_new, text = "New Class")


    # Saving the remaining dataset
    Class_0_remaining = Class0[:,size_of_set:Class0.shape[1]]
    Class_1_remaining = Class1[:,size_of_set:Class1.shape[1]]
    Class_2_remaining = Class2[:,size_of_set:Class2.shape[1]]
    Class_3_remaining = Class3[:,size_of_set:Class3.shape[1]]
    Class_4_remaining = Class4[:,size_of_set:Class4.shape[1]]
    Class_5_remaining = Class5[:,size_of_set:Class5.shape[1]]
    Class_6_remaining = Class6[:,size_of_set:Class6.shape[1]]
    Class_7_remaining = Class7[:,size_of_set:Class7.shape[1]]
    Class_8_remaining = Class8[:,size_of_set:Class8.shape[1]]
    Class_9_remaining = Class9[:,size_of_set:Class9.shape[1]]
    #Print_shape(Class_0_remaining, Class_1_remaining, Class_2_remaining, Class_3_remaining, Class_4_remaining, Class_5_remaining, Class_6_remaining, Class_7_remaining, Class_8_remaining, Class_9_remaining, text = "Remaining Class")


    Y_New = np.zeros((size_of_set*int(pd.Series(Y).nunique())), dtype = int)

    Y_New[0 : size_of_set] = 0
    Y_New[(size_of_set+1) : size_of_set*2] = 1
    Y_New[(size_of_set*2)+1 : size_of_set*3] = 2
    Y_New[(size_of_set*3)+1 : size_of_set*4] = 3
    Y_New[(size_of_set*4)+1 : size_of_set*5] = 4
    Y_New[(size_of_set*5)+1 : size_of_set*6] = 5
    Y_New[(size_of_set*6)+1 : size_of_set*7] = 6
    Y_New[(size_of_set*7)+1 : size_of_set*8] = 7
    Y_New[(size_of_set*8)+1 : size_of_set*9] = 8
    Y_New[(size_of_set*9)+1 : size_of_set*10] = 9

    Y_0_remaining = np.zeros((Class0.shape[1] - size_of_set), dtype = int)
    Y_1_remaining = np.zeros((Class1.shape[1] - size_of_set), dtype = int)
    Y_2_remaining = np.zeros((Class2.shape[1] - size_of_set), dtype = int)
    Y_3_remaining = np.zeros((Class3.shape[1] - size_of_set), dtype = int)
    Y_4_remaining = np.zeros((Class4.shape[1] - size_of_set), dtype = int)
    Y_5_remaining = np.zeros((Class5.shape[1] - size_of_set), dtype = int)
    Y_6_remaining = np.zeros((Class6.shape[1] - size_of_set), dtype = int)
    Y_7_remaining = np.zeros((Class7.shape[1] - size_of_set), dtype = int)
    Y_8_remaining = np.zeros((Class8.shape[1] - size_of_set), dtype = int)
    Y_9_remaining = np.zeros((Class9.shape[1] - size_of_set), dtype = int)

    Y_0_remaining[:] = 0
    Y_1_remaining[:] = 1
    Y_2_remaining[:] = 2
    Y_3_remaining[:] = 3
    Y_4_remaining[:] = 4
    Y_5_remaining[:] = 5
    Y_6_remaining[:] = 6
    Y_7_remaining[:] = 7
    Y_8_remaining[:] = 8
    Y_9_remaining[:] = 9

    X_train = np.concatenate((Class0_new, Class1_new, Class2_new, Class3_new, Class4_new, Class5_new, Class6_new, Class7_new, Class8_new, Class9_new), axis = 1)
    
    X_remaining = np.concatenate((Class_0_remaining, Class_1_remaining, Class_2_remaining, Class_3_remaining, Class_4_remaining, Class_5_remaining, Class_6_remaining, Class_7_remaining, Class_8_remaining, Class_9_remaining), axis = 1)
    
    Y_remaining = np.concatenate((Y_0_remaining, Y_1_remaining, Y_2_remaining, Y_3_remaining, Y_4_remaining, Y_5_remaining, Y_6_remaining, Y_7_remaining, Y_8_remaining, Y_9_remaining))

    S_X, S_Y = shuffle_Data_2D(X_train, Y_New)
    X_remaining, Y_remaining = shuffle_Data_2D(X_remaining, Y_remaining)
    # Partition training into training and development set
    X_train, X_devel = S_X[:, :-devel_size], S_X[:, -devel_size:]
    Y_train, Y_devel = S_Y[:-devel_size], S_Y[-devel_size:]
    
    #print("Shape of Training Set = " + str(X_train.shape))
    #print("Shape of Training Set Labels = " + str(Y_train.shape))
    #print("Shape of Validation Set = " + str(X_devel.shape))
    #print("Shape of Validation Set Labels= " + str(Y_devel.shape))
    #print("Shape of Remaining Set = " + str(X_remaining.shape))
    #print("Shape of Remaining Set Labels= " + str(Y_remaining.shape))
    
    
    return X_train, Y_train, X_devel, Y_devel, X_remaining, Y_remaining
    
        
def display_graph(Y):
    '''
    This function displays the number of 
    classes and displays the number of samples 
    in each class numerically and on a graph.
    '''
    df = pandas.Series(Y)
    df.groupby(df).size().plot(kind = 'bar')
    print("Class:  No: of Images")
    print(df.groupby(df).size())
    df.nunique()

def display_image(image_num, S_X, S_Y):
    '''
    This function displays the image along with its label:
    image_num = image number
    S_X = which image from the trainming dataset
    S_Y = label corresponding to that image.
    '''
    image = S_X[:,image_num]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print("Label = " + str(S_Y[image_num]))



def Class_Distribution_recall(X, Y, size_of_set, confused_classes):
    
    # Classifying the data into classwise
    toto = np.where(Y == 0)
    Class0 = X[:, toto]
    toto = np.where(Y == 1)
    Class1 = X[:, toto]
    toto = np.where(Y == 2)
    Class2 =X[:, toto]
    toto = np.where(Y == 3)
    Class3 =X[:, toto]
    toto = np.where(Y == 4)
    Class4 =X[:, toto]
    toto = np.where(Y == 5)
    Class5 =X[:, toto]
    toto = np.where(Y == 6)
    Class6 =X[:, toto]
    toto = np.where(Y == 7)
    Class7 =X[:, toto]
    toto = np.where(Y == 8)
    Class8 =X[:, toto]
    toto = np.where(Y == 9)
    Class9 =X[:, toto]
    
    # Removing the third Dimension
    Class0 = Class0[:,0,:]
    Class1 = Class1[:,0,:]
    Class2 = Class2[:,0,:]
    Class3 = Class3[:,0,:]
    Class4 = Class4[:,0,:]
    Class5 = Class5[:,0,:]
    Class6 = Class6[:,0,:]
    Class7 = Class7[:,0,:]
    Class8 = Class8[:,0,:]
    Class9 = Class9[:,0,:]
    #Print_shape(Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, text = "Main Class")

    Y_New = np.zeros((size_of_set*len(confused_classes)), dtype = int)
    
    def check_i (i, size_of_set):
        if i == 0:
            i = i+1
            New_Value = np.array(range(0,size_of_set))
            return i, New_Value
            
        if i==1:
            i=i+1
            New_Value = np.array(range(size_of_set,size_of_set*2))
            return i, New_Value
        
        if i==2:
            i=i+1
            New_Value = np.array(range(size_of_set*2,size_of_set*3))
            return i, New_Value
        
        if i==3:
            i=i+1
            New_Value = np.array(range(size_of_set*3,size_of_set*4))
            return i, New_Value
        
        if i==4:
            i=i+1
            New_Value =  np.array(range(size_of_set*4,size_of_set*5))
            return i, New_Value
        
        if i==5:
            i=i+1
            New_Value = np.array(range(size_of_set*5,size_of_set*6))
            return i, New_Value
        
        if i==6:
            i=i+1
            New_Value =np.array(range(size_of_set*6,size_of_set*7))
            return i, New_Value
        
        if i==7:
            i=i+1
            New_Value =np.array(range(size_of_set*7,size_of_set*8))
            return i, New_Value
        
        if i==8:
            i=i+1
            New_Value = np.array(range(size_of_set*8,size_of_set*9))
            return i, New_Value
        
        if i==9:
            i=i+1
            New_Value = np.array(range(size_of_set*9,size_of_set*10))
            return i, New_Value
    
    X_train=[]
    X_train = np.array(X_train)
    # Dividing the dataset
    i=0
    
    if np.any(np.array(confused_classes)==0):
        Class0_new = Class0[:,0:size_of_set]
        if i==0:
            X_train = Class0_new
        else:
            X_train = np.hstack((X_train, Class0_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 0
    
    if np.any(np.array(confused_classes)==1):
        Class1_new = Class1[:,0:size_of_set]
        if i==0:
            X_train = Class1_new
        else:
            X_train = np.hstack((X_train, Class1_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 1
        
    if np.any(np.array(confused_classes)==2):
        Class2_new = Class2[:,0:size_of_set]
        if i==0:
            X_train = Class2_new
        else:
            X_train = np.hstack((X_train, Class2_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 2
        
    if np.any(np.array(confused_classes)==3):
        Class3_new = Class3[:,0:size_of_set]
        if i==0:
            X_train = Class3_new
        else:
            X_train = np.hstack((X_train, Class3_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 3
        
    if np.any(np.array(confused_classes)==4):
        Class4_new = Class4[:,0:size_of_set]
        if i==0:
            X_train = Class4_new
        else:
            X_train = np.hstack((X_train, Class4_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 4
        
    if np.any(np.array(confused_classes)==5):
        Class5_new = Class5[:,0:size_of_set]
        if i==0:
            X_train = Class5_new
        else:
            X_train = np.hstack((X_train, Class5_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 5
        
    if np.any(np.array(confused_classes)==6):
        Class6_new = Class6[:,0:size_of_set]
        if i==0:
            X_train = Class6_new
        else:
            X_train = np.hstack((X_train, Class6_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 6
       
    if np.any(np.array(confused_classes)==7):
        Class7_new = Class7[:,0:size_of_set]
        if i==0:
            X_train = Class7_new
        else:
            X_train = np.hstack((X_train, Class7_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 7
        
    if np.any(np.array(confused_classes)==8):
        Class8_new = Class8[:,0:size_of_set]
        if i==0:
            X_train = Class8_new
        else:
            X_train = np.hstack((X_train, Class8_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 8
        
    if np.any(np.array(confused_classes)==9):
        Class9_new = Class9[:,0:size_of_set]
        if i==0:
            X_train = Class9_new
        else:
            X_train = np.hstack((X_train, Class9_new))
        i, hello =  check_i(i, size_of_set)
        Y_New[hello] = 9

    X_train = np.asarray(X_train)
    Y_New = np.asarray(Y_New)
    print("Shape of Training Set X_Train = " + str(X_train.shape))
    print("Shape of Training Set Labels = " + str(Y_New.shape))
    
    S_X, S_Y = shuffle_Data_2D(X_train, Y_New)
    
    #print("Shape of Training Set = " + str(X_train.shape))
    #print("Shape of Training Set Labels = " + str(Y_New.shape))
    
    #print("Shape of Training Set after shuf= " + str(S_X.shape))
    #print("Shape of Training Set Labels after shuf= " + str(S_Y.shape))

    return S_X, S_Y


def Class_Distribution_x_y_remaining(X, Y, confused_classes):
    
    # Classifying the data into classwise
    toto = np.where(Y == 0)
    Class0 = X[:, toto]
    toto = np.where(Y == 1)
    Class1 = X[:, toto]
    toto = np.where(Y == 2)
    Class2 =X[:, toto]
    toto = np.where(Y == 3)
    Class3 =X[:, toto]
    toto = np.where(Y == 4)
    Class4 =X[:, toto]
    toto = np.where(Y == 5)
    Class5 =X[:, toto]
    toto = np.where(Y == 6)
    Class6 =X[:, toto]
    toto = np.where(Y == 7)
    Class7 =X[:, toto]
    toto = np.where(Y == 8)
    Class8 =X[:, toto]
    toto = np.where(Y == 9)
    Class9 =X[:, toto]
    
    # Removing the third Dimension
    Class0 = Class0[:,0,:]
    Class1 = Class1[:,0,:]
    Class2 = Class2[:,0,:]
    Class3 = Class3[:,0,:]
    Class4 = Class4[:,0,:]
    Class5 = Class5[:,0,:]
    Class6 = Class6[:,0,:]
    Class7 = Class7[:,0,:]
    Class8 = Class8[:,0,:]
    Class9 = Class9[:,0,:]
    #Print_shape(Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, text = "Main Class")

    y_label = [0,1,2,3,4,5,6,7,8,9]
    y = []
    for i in range(10):
        isinthelist = 0
        for j in range(len(confused_classes)):
            if y_label[i] == confused_classes[j]:
                isinthelist = 1
        if isinthelist == 0:
            y.append(y_label[i])
    y_remaining_classes = list(dict.fromkeys(y))
    print('remaining classes:',y_remaining_classes)

    a=0
    b=0
    y_train = Y
    y_new = []
    x_new=[]
    x_new = np.array(x_new)
    if np.any(np.array(y_remaining_classes) == 0):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 0]))
            a = a + Class0.shape[1]
            x_new = Class0
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 0]))
            x_new = np.hstack((x_new, Class0))
            #print(x_new.shape)
            a = a + Class0.shape[1]

    if np.any(np.array(y_remaining_classes) == 1):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 1]))
            a = a + Class1.shape[1]
            x_new = Class1
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 1]))
            x_new = np.hstack((x_new, Class1))
            #print(x_new.shape)
            a = a + Class1.shape[1]

    if np.any(np.array(y_remaining_classes) == 2):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 2]))
            a = a + Class2.shape[1]
            x_new = Class2
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 2]))
            x_new = np.hstack((x_new, Class2))
            #print(x_new.shape)
            a = a + Class2.shape[1]

    if np.any(np.array(y_remaining_classes) == 3):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 3]))
            a = a + Class3.shape[1]
            x_new = Class3
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 3]))
            a = a + Class3.shape[1]
            x_new = np.hstack((x_new, Class3))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 4):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 4]))
            a = a + Class4.shape[1]
            x_new = Class4
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 4]))
            a = a + Class4.shape[1]
            x_new = np.hstack((x_new, Class4))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 5):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 5]))
            a = a + Class5.shape[1]
            x_new = Class5
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 5]))
            a = a + Class5.shape[1]
            x_new = np.hstack((x_new, Class5))
            #print(x_new.shape)
            
            
    if np.any(np.array(y_remaining_classes) == 6):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 6]))
            a = a + Class6.shape[1]
            x_new = Class6
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 6]))
            a = a + Class6.shape[1]
            x_new = np.hstack((x_new, Class6))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 7):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 7]))
            a = a + Class7.shape[1]
            x_new = Class7
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 7]))
            a = a + Class7.shape[1]
            x_new = np.hstack((x_new, Class7))
            #print(x_new.shape)
                
            
    if np.any(np.array(y_remaining_classes) == 8):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 8]))
            a = a + Class8.shape[1]
            x_new = Class8
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 8]))
            a = a + Class8.shape[1]
            x_new = np.hstack((x_new, Class8))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 9):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 9]))
            a = a + Class9.shape[1]
            x_new = Class9
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 9]))
            a = a + Class9.shape[1]
            x_new = np.hstack((x_new, Class9))
            #print(x_new.shape)
                
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)
    
    S_X, S_Y = shuffle_Data_2D(x_new, y_new)
    
    x_new = np.transpose(S_X)
    
    #print(a) 
    #print('x_new size:', x_new.shape)
    #print('y_new size:', y_new.shape)
    
    return x_new, S_Y




def Class_Distribution_x_y_test_confused(X, Y, confused_classes):
    
    # Classifying the data into classwise
    toto = np.where(Y == 0)
    Class0 = X[:, toto]
    toto = np.where(Y == 1)
    Class1 = X[:, toto]
    toto = np.where(Y == 2)
    Class2 =X[:, toto]
    toto = np.where(Y == 3)
    Class3 =X[:, toto]
    toto = np.where(Y == 4)
    Class4 =X[:, toto]
    toto = np.where(Y == 5)
    Class5 =X[:, toto]
    toto = np.where(Y == 6)
    Class6 =X[:, toto]
    toto = np.where(Y == 7)
    Class7 =X[:, toto]
    toto = np.where(Y == 8)
    Class8 =X[:, toto]
    toto = np.where(Y == 9)
    Class9 =X[:, toto]
    
    # Removing the third Dimension
    Class0 = Class0[:,0,:]
    Class1 = Class1[:,0,:]
    Class2 = Class2[:,0,:]
    Class3 = Class3[:,0,:]
    Class4 = Class4[:,0,:]
    Class5 = Class5[:,0,:]
    Class6 = Class6[:,0,:]
    Class7 = Class7[:,0,:]
    Class8 = Class8[:,0,:]
    Class9 = Class9[:,0,:]
    #Print_shape(Class0, Class1, Class2, Class3, Class4, Class5, Class6, Class7, Class8, Class9, text = "Main Class")

    y = []
    y_remaining_classes = confused_classes
    print('confused classes:',y_remaining_classes)

    a=0
    b=0
    y_train = Y
    y_new = []
    x_new=[]
    x_new = np.array(x_new)
    if np.any(np.array(y_remaining_classes) == 0):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 0]))
            a = a + Class0.shape[1]
            x_new = Class0
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 0]))
            x_new = np.hstack((x_new, Class0))
            #print(x_new.shape)
            a = a + Class0.shape[1]

    if np.any(np.array(y_remaining_classes) == 1):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 1]))
            a = a + Class1.shape[1]
            x_new = Class1
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 1]))
            x_new = np.hstack((x_new, Class1))
            #print(x_new.shape)
            a = a + Class1.shape[1]

    if np.any(np.array(y_remaining_classes) == 2):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 2]))
            a = a + Class2.shape[1]
            x_new = Class2
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 2]))
            x_new = np.hstack((x_new, Class2))
            #print(x_new.shape)
            a = a + Class2.shape[1]

    if np.any(np.array(y_remaining_classes) == 3):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 3]))
            a = a + Class3.shape[1]
            x_new = Class3
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 3]))
            a = a + Class3.shape[1]
            x_new = np.hstack((x_new, Class3))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 4):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 4]))
            a = a + Class4.shape[1]
            x_new = Class4
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 4]))
            a = a + Class4.shape[1]
            x_new = np.hstack((x_new, Class4))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 5):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 5]))
            a = a + Class5.shape[1]
            x_new = Class5
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 5]))
            a = a + Class5.shape[1]
            x_new = np.hstack((x_new, Class5))
            #print(x_new.shape)
            
            
    if np.any(np.array(y_remaining_classes) == 6):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 6]))
            a = a + Class6.shape[1]
            x_new = Class6
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 6]))
            a = a + Class6.shape[1]
            x_new = np.hstack((x_new, Class6))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 7):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 7]))
            a = a + Class7.shape[1]
            x_new = Class7
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 7]))
            a = a + Class7.shape[1]
            x_new = np.hstack((x_new, Class7))
            #print(x_new.shape)
                
            
    if np.any(np.array(y_remaining_classes) == 8):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 8]))
            a = a + Class8.shape[1]
            x_new = Class8
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 8]))
            a = a + Class8.shape[1]
            x_new = np.hstack((x_new, Class8))
            #print(x_new.shape)

    if np.any(np.array(y_remaining_classes) == 9):
        if b == 0:
            y_new = np.hstack((y_new, y_train[y_train == 9]))
            a = a + Class9.shape[1]
            x_new = Class9
            #print(x_new.shape)
            b = b + 1
        else:
            y_new = np.hstack((y_new, y_train[y_train == 9]))
            a = a + Class9.shape[1]
            x_new = np.hstack((x_new, Class9))
            #print(x_new.shape)
                
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)
    S_X, S_Y = shuffle_Data_2D(x_new, y_new)
    
    x_new = np.transpose(S_X)
    #print(a) 
    #print('x_new size:', x_new.shape)
    #print('y_new size:', y_new.shape)
    
    return x_new,S_Y





class TimeTestAccHistory(keras.callbacks.Callback):
    def __init__(self, x_test_call, y_test, y_test_call):
        super(TimeTestAccHistory, self).__init__()
        self.x_test_call = x_test_call
        self.y_test = y_test
        self.y_test_call = y_test_call

    def on_train_begin(self, logs={}):
        self.times = []
        self.test_acc = []
        self.avg_conf = []
        self.max_miscl_num = []
        self.conf_class_median = []
        self.std_conf = []
        self.most_conf_class_1 = []
        self.most_conf_class_2 = []
        self.sum_of_confused_classes = []

    def on_epoch_begin(self, batch, logs={}):
        self.start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.start)        
        test_accuracy = self.model.evaluate(self.x_test_call, self.y_test_call)
        self.test_acc.append(test_accuracy[1])
        
        rounded_pred_call = self.model.predict_classes(self.x_test_call, batch_size = 32, verbose = 0)
        cm = confusion_matrix(rounded_pred_call, self.y_test)
        np.fill_diagonal(cm, 0)
        Total_misclassified_images = cm.sum()
        self.sum_of_confused_classes.append(Total_misclassified_images)
        
        percentage = self.y_test_call.shape[1] / 10
        remaining = (self.y_test_call.shape[1] - np.int(percentage))*10
        Average_Confusion = Total_misclassified_images/remaining
        self.avg_conf.append(Average_Confusion)
        
        maximum_misclassified_images = cm.max()
        self.max_miscl_num.append(maximum_misclassified_images)
        
        confused_classes=[]
        for i in range(len(cm)):
            for j in range (len(cm)):
                if i != j and cm[i,j] >= maximum_misclassified_images:
                    #print('Class ' + str(i) + ' and Class ' + str(j) + ' have high confusion rate')
                    confused_classes.append(i)
                    confused_classes.append(j)
                    
        self.most_conf_class_1.append(confused_classes[0])
        self.most_conf_class_2.append(confused_classes[1])
        
        con_median = np.median(cm)
        self.conf_class_median.append(con_median)
        
        std_conf = np.std(cm)
        self.std_conf.append(std_conf)
        
        
        
        
def train_model_mnist_call(model, train, val, test, num_classes, input_shape, batch_size, epochs, lr, callbacks, time_history, mode, hist_csv_file):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    Dev_X = val[0].reshape((val[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #print('x_train shape:', x_train.shape)
    #print('Dev_X Shape:', Dev_X.shape)
    #print("X_test Shape:", x_test.shape)
    #print(x_train.shape[0], 'train samples')
    #print(Dev_X.shape[0], 'validation samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    optimizer = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizer,
                  metrics=['accuracy'])

    t = now()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y), callbacks = callbacks)
    history.history['time'] = time_history.times
    history.history['test_acc'] = time_history.test_acc
    history.history['avg_conf'] = time_history.avg_conf
    history.history['max_miscl_num'] = time_history.max_miscl_num
    history.history['conf_class_median'] = time_history.conf_class_median
    history.history['std_conf'] = time_history.std_conf
    history.history['most_conf_class_1'] = time_history.most_conf_class_1
    history.history['most_conf_class_2'] = time_history.most_conf_class_2
    history.history['sum_of_confused_classes'] = time_history.sum_of_confused_classes
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    with open(hist_csv_file, mode=mode, newline = '\n') as f:
        #print(hist_df.head())
        hist_df.to_csv(f, index= False, header=mode=='w')
    
    print("*******************************************")
    print('Training time: %s' % (now() - t))
    #print('x_test shape:', x_test.shape)
    #print('y_test shape:', y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("*******************************************")
    #print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print("*******************************************")
    
    
    
    
    
    
def train_model_fmnist_call(model, train, val, test, num_classes, input_shape, batch_size, epochs, lr, callbacks, time_history, mode, hist_csv_file):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    Dev_X = val[0].reshape((val[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    Dev_X = Dev_X.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    #print('x_train shape:', x_train.shape)
    #print('Dev_X Shape:', Dev_X.shape)
    #print("X_test Shape:", x_test.shape)
    #print(x_train.shape[0], 'train samples')
    #print(Dev_X.shape[0], 'validation samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    Dev_Y = keras.utils.to_categorical(val[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    optimizer = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizer,
                  metrics=['accuracy'])

    t = now()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(Dev_X, Dev_Y), callbacks = callbacks)
    history.history['time'] = time_history.times
    history.history['test_acc'] = time_history.test_acc
    history.history['avg_conf'] = time_history.avg_conf
    history.history['max_miscl_num'] = time_history.max_miscl_num
    history.history['conf_class_median'] = time_history.conf_class_median
    history.history['std_conf'] = time_history.std_conf
    history.history['most_conf_class_1'] = time_history.most_conf_class_1
    history.history['most_conf_class_2'] = time_history.most_conf_class_2
    history.history['sum_of_confused_classes'] = time_history.sum_of_confused_classes
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    with open(hist_csv_file, mode=mode, newline = '\n') as f:
        #print(hist_df.head())
        hist_df.to_csv(f, index= False, header=mode=='w')
    
    print("*******************************************")
    print('Training time: %s' % (now() - t))
    #print('x_test shape:', x_test.shape)
    #print('y_test shape:', y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("*******************************************")
    #print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print("*******************************************")