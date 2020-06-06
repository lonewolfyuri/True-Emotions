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

class fer2013():
    def __init__(self,path=r'fer2013.pkl'):
        dF = pd.read_pickle(path)
        self.emotion = dF['emotion']
        self.pixels = dF['pixels']
        self.Usage = dF['Usage']
    def plot_image(self,imageNumber,label=-1,label_string=True,ax=None,cmap='gray'):
        if label == -1:
            label = self.emotion[imageNumber]
        if label_string == True:
            label = self.label_key(label)
        if ax == None:
            plt.figure()
            plt.imshow(self.pixels[imageNumber],cmap=cmap)
            plt.title(label)
            plt.axis('off')
        else:
            ax.imshow(self.pixels[imageNumber],cmap=cmap)
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

    def fit_transform(self):
        array = np.zeros((self.pixels.shape[0],48*48))
        for ii in range(len(self.pixels)):
            array[ii] = self.pixels[ii].reshape(48*48)
        self.scaler = StandardScaler()
        scaled_array = self.scaler.fit_transform(array)
        print(scaled_array - array)
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
        indices = np.linspace(0,len(self.pixels)-1,len(self.pixels))
        if seed == None:
            np.random.shuffle(indices)
        else:
            randomState = np.random.RandomState(seed=seed)
            randomState.shuffle(indices)
        print(indices)
        self.emotion[indices] = self.emotion
        self.pixels[indices] = self.pixels
        self.Usage[indices] = self.Usage

if __name__ == '__main__':
    data = fer2013()
    data.shuffle()
    plt.close('all')
    for ii in range(2):
        fig, ax = plt.subplots(ncols=3)
        data.plot_image(ii,ax=ax[0])
        data.fit_transform()
        data.plot_image(ii,ax=ax[1])
        data.inverse_transform()
        data.plot_image(ii,ax=ax[2])