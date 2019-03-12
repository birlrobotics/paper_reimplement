# -*- coding: utf-8 -*-
"""
Cluster the probability distribution use Bhattacharyya Distances D_psi or D_phi,\
In the paper, the max distance eps_psi or eps_phi is 0.05cm
Step 1. Use Maximum Likelihood to find the best mean and covariance matrix
Step 2. Use multivariate normal distribution to compute the corresponding distribution
Step 3. Redefine the input distribution by compare the Bhattacharyya Distance
"""

# Author: Jim Huang <huangjiancong863@gmail.com>
#         
#
# License: MIT License

import numpy as np

def mean_h(data):
    """Using Maximum Likelihood Estimate to find the mean

    Parameters
    ----------
    data : array_like
        Like ([1, 2, 3], [2, 3, 4],......, [4, 3, 1])

    Return
    ------
    mean_set : array_like
        The mean of input set
    """
    # mean_set = [] 
    # a = []   
    # for i in range(len(data[0])):
    #     for j in range(len(data)):
    #         a.append(data[j][i])
    #     mean = np.mean(a)
    #     mean_set.append(mean)
    # return mean_set
    mean_set = np.mean(data, axis=0) # Calculate the mean of each column
    return mean_set


def cov_h(data):
    """Using Maximum Likelihood Estimate to find the Covariance Matrix
        and Diagonalization

    Parameters
    ----------
    data : array_like
        Like ([1, 2, 3], [2, 3, 4],......, [4, 3, 1])
    mean_set : array_like
        The mean of the input set

    Return
    ------
    cov_matrix : array_like
        The diagonal covariance matrix of input set
    """
    c = np.cov(data.T) 
    eigen_vals, eigen_vect = np.linalg.eig(c)
    cov_matrix = np.diag(eigen_vals)
    return cov_matrix

def Redefine
    pass


    