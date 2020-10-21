# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:36:20 2020

@author: Vaughn Weirens
"""
# Only run this if you want to re-generate the neural network.

# Import necessary modules
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import os

# Delete file if it exists
if os.path.isfile('Weirens_Neural_Network.sav'):
    os.remove('Weirens_Neural_Network.sav')

# Load in Training Data
x = np.genfromtxt('x.csv', delimiter=',')
y = np.genfromtxt('y.csv', delimiter=',')
z = np.genfromtxt('z.csv', delimiter=',')

xy = np.array([x, y]).transpose()

xy_train, xy_test, z_train, z_test = train_test_split(xy, z, random_state=1)

regr = MLPRegressor(random_state=1, max_iter=500,
                    hidden_layer_sizes=(500, 500, 500)).fit(xy_train, z_train)

pickle.dump(regr, open('Weirens_Neural_Network.sav', 'wb'))

coeff_deter = regr.score(xy_test, z_test)
print('The coefficient of determination for the initial test is: ',
      coeff_deter)
