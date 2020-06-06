# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:42:56 2020

@author: Matthew
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
class fer2013():
    def __init__(self,path=r'fer2013.pkl'):
        dF = pd.read_pickle(path)

        #Loads images and arranges them into a arary of size nxmxmx1
        self.pixels = np.stack(dF['pixels'])
        self.pixels = np.reshape(self.pixels,(len(self.pixels),len(self.pixels[0]),len(self.pixels[0][0]),1))
        self.Usage = dF['Usage'].values
        self.emotion = dF['emotion'].values
        self.emotion = np.reshape(self.emotion,(len(self.emotion),1))
        self.targets = np.zeros((len(self.emotion),7))

        for ii in range(len(self.emotion)):
            self.targets[ii,self.emotion[ii]] = 1

    def augment(self,augment_type='flip'):
        if augment_type == 'flip':
            aug = self.pixels*1.0
            for ii in range(len(aug)):
                aug[ii] = np.fliplr(aug[ii][:,:,0]).reshape((len(aug[0]),len(aug[0][0]),1))
            self.pixels = np.vstack((self.pixels,aug))
            self.targets = np.vstack((self.targets,self.targets))
            self.Usage = np.hstack((self.Usage,self.Usage))
            self.emotion = np.vstack((self.emotion,self.emotion))

        else:
            print('Invalid augment_type')
    def plot_image(self,imageNumber,label=-1,label_string=True,ax=None,cmap='gray'):
        if label == -1:
            label = self.emotion[imageNumber][0]
        if label_string == True:
            label = self.label_key(label)
        if ax == None:
            plt.figure()
            plt.imshow(self.pixels[imageNumber][:,:,0],cmap=cmap)
            plt.title(label)
            plt.axis('off')
        else:
            ax.imshow(self.pixels[imageNumber][:,:,0],cmap=cmap)
            ax.set_title(label)
            ax.axis('off')

    def label_key(self,label_num):
        if label_num == 0:
            label_string = 'Angry'
        elif label_num == 1:
            label_string = 'Disgust'
        elif label_num == 2:
            label_string = 'Fear'
        elif label_num == 3:
            label_string = 'Happy'
        elif label_num == 4:
            label_string = 'Sad'

        elif label_num == 5:
            label_string = 'Surprised'
        elif label_num == 6:
            label_string = 'Neutral'
        else:
            print('Unknown label number')
            label_string ='Invalid'
        return label_string

    def normalize(self):
        self.pixels = self.pixels/255.0

    def fit_transform(self):
        array = np.zeros((self.pixels.shape[0],48*48))
        for ii in range(len(self.pixels)):
            array[ii] = self.pixels[ii].reshape(48*48)
        self.scaler = StandardScaler()
        scaled_array = self.scaler.fit_transform(array)
        for ii in range(len(self.pixels)):
            self.pixels[ii] = scaled_array[ii].reshape(48,48)

    def inverse_transform(self):
        array = np.zeros((self.pixels.shape[0],48*48))
        for ii in range(len(self.pixels)):
            array[ii] = self.pixels[ii].reshape(48*48)
        scaled_array = self.scaler.inverse_transform(array)
        for ii in range(len(self.pixels)):
            self.pixels[ii] = scaled_array[ii].reshape(48,48)

    def shuffle(self,seed=None):
        indices = np.linspace(0,len(self.pixels)-1,len(self.pixels)).astype(int)
        if seed == None:
            np.random.shuffle(indices)
        else:
            randomState = np.random.RandomState(seed=seed)
            randomState.shuffle(indices)
        self.emotion[indices] = self.emotion
        self.pixels[indices] = self.pixels
        self.Usage[indices] = self.Usage
        self.targets[indices] = self.targets

    def split_data(self,test_size=0.2,seed=None):
        if seed == None:
            seed = time.time()
        x_train,x_test,y_train,y_test = train_test_split(self.pixels,self.targets,test_size=test_size,random_state=int(seed))
        return x_train,y_train,x_test,y_test
if __name__ == '__main__':
    data = fer2013()
    data.shuffle(seed=10)

    data.augment()
    print(data.Usage.shape)
#    plt.close('all')
#    for ii in range(2):
#        fig, ax = plt.subplots(ncols=3)
#        data.plot_image(ii,ax=ax[0])
#        #data.fit_transform()
#        data.plot_image(ii,ax=ax[1])
#        #data.inverse_transform()
#        data.plot_image(ii,ax=ax[2])
#        fig, ax = plt.subplots(ncols=3)
#        data.plot_image(int(ii+len(data.targets)),ax=ax[0])
#        #data.fit_transform()
#        data.plot_image(int(ii+len(data.targets)),ax=ax[1])
#        #data.inverse_transform()
#        data.plot_image(int(ii+len(data.targets)),ax=ax[2])