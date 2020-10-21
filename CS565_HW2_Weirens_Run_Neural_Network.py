# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:36:57 2020

@author: Vaughn Weirens
"""

import numpy as np
import pickle

xname = input('Please enter the filename for the "x" input: ')
yname = input('Please enter the filename for the "y" input: ')

x = np.genfromtxt(xname, delimiter=',')
y = np.genfromtxt(yname, delimiter=',')

xy = np.array([x, y]).transpose()

regr = pickle.load(open('Weirens_Neural_Network.sav', 'rb'))

z_predicted = regr.predict(xy)
np.savetxt('z_predicted.csv', z_predicted, delimiter=',')
