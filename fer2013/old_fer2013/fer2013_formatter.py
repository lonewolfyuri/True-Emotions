# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:46:17 2020

@author: Matthew
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py


data = pd.read_csv(r'fer2013.csv')



for ii in range(data.shape[0]):
    data['pixels'][ii] = np.asarray(data['pixels'][ii].split(' ')).astype(float).reshape(48,48)

data.to_pickle(r'fer2013.pkl')

