# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:56:51 2020

@author: Vaughn Weirens
"""

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import numpy as np

n = 5

gm_data = np.genfromtxt('p2-data.csv', delimiter=',')

gm_train, gm_test = train_test_split(gm_data)

estimators = {cov_type: GaussianMixture(n_components=n,
                                        covariance_type=cov_type,
                                        max_iter=100, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

n_estimators = len(estimators)

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.fit(gm_train)
    print('The mean positions determined by the ', name, ' estimator are: \n',
          estimator.means_, '\n')
    print('The covariances determined by the ', name, 'estimator are: \n',
          estimator.covariances_, '\n')