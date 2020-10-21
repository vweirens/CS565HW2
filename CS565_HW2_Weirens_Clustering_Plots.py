# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:17:36 2020

@author: Vaughn Weirens
"""

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

colors = ['red', 'turquoise', 'navy', 'darkorange', 'yellow']
n = len(colors)


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


gm_data = np.genfromtxt('p2-data.csv', delimiter=',')

gm_train, gm_test = train_test_split(gm_data)

estimators = {cov_type: GaussianMixture(n_components=n,
                                        covariance_type=cov_type,
                                        max_iter=100, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

n_estimators = len(estimators)

plt.figure(figsize=(3*n_estimators//2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.fit(gm_train)
    
    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)
    
    plt.scatter(gm_data[:, 0], gm_data[:, 1], marker='x', color='black')

    plt.title(name)
    
    print('The mean positions determined by the ', name, ' estimator are: \n',
          estimator.means_, '\n')
    print('The covariances determined by the ', name, 'estimator are: \n',
          estimator.covariances_, '\n')

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

plt.show()
