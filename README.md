# CS565HW2

## Overview
Homework submission for assignment 2: Working with Neural Networks for Clustering and Regression algorithms

This repository contains two separate methods, a Clustering algorithm for a provided data set and a Multivariate Regression algorithm with a provided pre-generated neural network.

The code in this repository was constructed with assistance from the documentation at scikit-learn.org, particularly the following articles:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py

## Running the Regression Function

If the user wishes to re-generate the Neural Network, they may call `py -m CS565_HW2_Weirens_Generate_Neural_Network`, which will delete the saved Weirens_Neural_Network.sav neural network file and will generate a new one based on the provided data

NOTE: DO NOT RUN THIS FILE UNLESS YOU WISH TO GENERATE A NEW NEURAL NETWORK AS IT WILL OVERRIDE THE PRIOR ONE

To test the trained network, the user may enter `py -m CS565_HW2_Weirens_Run_Neural_Network`, after which the program will prompt the user to enter two file names, one for the x variable and one for the y variable. After this, the program will output the predicted data to z_predicted.csv.

## Running the Clustering Function

The clustering algorithm may be run by calling 'py -m CS565_HW2_Weirens_Clustering` from the command line. This will then output the means and covariances for the estimated gaussian mixtures for each cluster.

Another file, CS565_HW2_Weirens_Clustering_Plots includes matplotlib plots, although this may function better if run from an anaconda IDE, as it was noted during testing that matplotlib.pyplot did not always output to command line, but worked perfectly in the Anaconda Spyder IDE.

Gaussian Mixtures were selected as the algorithm to determine these clusters as the problem statement wished for a mean and covariance value for each cluster to be provided, and using gaussian mixtures to define clusters of data is the most straightforward way to accomplish this. K-Means was considered for this algorithm, but it was determined that it would be more difficult to determine a covariance for each of the K clusters.

## Clustering Testing Notes

Both python scripts pull in clustering data from p2-data, and through testing and visualization it was determined that 5 clusters generates the best results. Below is justification for why 5 clusters appears to be the best, with comparisons for other amounts of clusters provided:

| Number of Clusters | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| Results | A single cluster does not accurately reflect the data, as there is significant separation between points | 2 clusters looks better, as some are quite separated from the y axis, but there are many outliers that are not sufficiently encompased by 2 clusters. | There is still an unclassified grouping of data that is not properly assigned with only 3 clusters. | This looks quite close, but there are still some groups that appear to have some separation and are joined together. Notably, the mean for one of the clusters appears to be positioned between two groups of data. | All of the groups of data appear to be properly assigned. Each cluster has a very clear mean and covariance that represents the majority of the data in its area. | When adding clusters over 5, some of the previously single groups appear to be split in an unnecessary way. Notably, one of the largest clusters is split into two from 5-6 because it is simply trying to find more ways to differentiate, even though the split cluster is still similar enough to be considered one cluster. | Same as 6, instead the largest cluster is split into 3, rather than 2 clusters. | 3 clusters from the results in the 5 cluster simuation are now split into 2, the clusters are becoming increasingly divided | The clusters continue to be increasingly and unnecessarily divided. | The clusters begin to overlap unnecessarily to accommodate the specified number.

Adding more clusters would only serve to overclassify the system. It was determined through these tests that 5 clusters fit the dataset most accurately.
