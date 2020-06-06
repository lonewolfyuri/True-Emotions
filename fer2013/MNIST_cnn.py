# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:47:14 2020

@author: Matthew
"""
# based off of https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# %%
from numpy import mean
from numpy import std
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
import keras
from fer2013_loader import fer2013
from keras.utils.vis_utils import plot_model
from keras.models import load_model

import time

def mnist_data():
# load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel

    #making sure data is correct dimensions
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    #making sure targets are categorical
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX,trainY, testX,testY

def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
# define cnn model
def define_model(input_shape=(28,28,1),output_shape=10):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(output_shape, activation='softmax'))
    # compile model
    opt = SGD(lr=0.005, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# %%
####Uncomment to compare mnist and fer2013 dataset
#trainX,trainY, testX,testY = mnist_data()
#data = fer2013()
#data.shuffle(seed=10)
#x_train,x_test,y_train,y_test = data.split_data()
#
#print(x_train.shape,trainX.shape)
#print(y_train.shape,trainY.shape)


# %%
#uncommdt for mnist dataset
#trainX,trainY, testX,testY = mnist_data()
#train_norm,test_norm = prep_pixels(trainX,testX)

# %%
#uncomment for fer2013 dataset

data = fer2013()
data.augment()
data.shuffle(seed=10)
data.normalize()
trainX,trainY, testX,testY = data.split_data(seed=11)
train_norm, test_norm = trainX,testX
# %%
startTime = time.time()

model = define_model(input_shape=trainX[0].shape,output_shape=trainY.shape[1])
plot_model(model, to_file='models/model_%f.png'%startTime, show_shapes=True, show_layer_names=True)
print('Data Size/num. parameters: %0.3f'%(len(train_norm)/model.count_params()))
# %%
history = model.fit(train_norm, trainY, epochs=20, batch_size=32, validation_split=0.2, verbose=1,shuffle=True)
model.save('models/model_%f.h5'%startTime)
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# %%

shuffledY = testY*1.0
randomState = np.random.RandomState(seed=12)
randomState.shuffle(shuffledY)

results = model.evaluate(test_norm, testY, batch_size=128)
print('test loss, test acc:', results)
pred = model.predict(test_norm)
squareSize = 5
counter = 0

for kk in range(5):
    fig, ax = plt.subplots(ncols=squareSize,nrows=squareSize,figsize = (15,10))

    for ii in range(squareSize):
        for jj in range(squareSize):
            if np.argmax(testY[counter])==np.argmax(pred[counter]):
                color='blue'
            else:
                color='darkred'

            ax[ii,jj].imshow(testX[counter][:,:,0],cmap='gray')
            ax[ii,jj].set_title('T: %s\nP: %s  P. Val: %0.2f'%(data.label_key(np.argmax(testY[counter])),data.label_key(np.argmax(pred[counter])),np.max(pred[counter])),color=color)
            ax[ii,jj].axis('off')
            counter +=1

    plt.tight_layout()
    plt.savefig('%05d.png'%kk)
    plt.close()
# %%
print('FER2013 Class Breakdown')
binnedY = np.sum(testY,axis=0)
binnedPred = np.sum(pred,axis=0)
setSize = len(testY)
for ii in range(len(testY[0])):
    print(data.label_key(ii) + ' %0.2f '%(binnedY[ii]/setSize))
print('Errors')
for ii in range(len(testY[0])):
    print(data.label_key(ii) + ' %0.3f '%(np.abs((binnedY[ii]-binnedPred[ii]))/binnedY[ii]))