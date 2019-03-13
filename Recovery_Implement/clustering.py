# -*- coding: utf-8 -*-
"""
Cluster the probability distribution use Bhattacharyya Distances D_psi or D_phi,\
In the paper, the max distance eps_psi or eps_phi is 0.05cm
"""

# Author: Jim Huang <huangjiancong863@gmail.com>
#         
#
# License: MIT License

def clustered_batch(classifications, samples):
    """Create a set which in the same clusters

    Parameters
	----------
	classifications : array_like
		The clustered number set after executed dbscan.py
	samples : array_like
		The original samples waiting the clustering.

	Return
	------
	cb : list
		The clustered samples' batchs.
    
    Example
    -------
    >>> from dbscan_bhatt import DBSCAN
    >>> import numpy as np
    >>> X = np.arraynp.array([[1, 2, 1], [2, 2, 2], [2, 3, 3], 
    ...               [8, 7, 7], [8, 8, 8], [25, 80, 50]])
    >>> clustering = DBSCAN(X, 3, 2)
    >>> from clustering import clustered_batch    
    >>> clustered_samples = clustered_batch(clustering, X)
    >>> clustered_samples
    >>> [[array([1, 2]), array([2, 2]), array([2, 3])], [array([8, 7]), 
    ...                array([8, 8])], [array([25, 80])]]
    """

    cb = []
    A = []
    # search the index with same value
    for i in set(classifications):
        address_index = [x for x in range(len(classifications)) if classifications[x] == i]
        for j in address_index:
            A.append(samples[j]).tolist
        cb.append(A)
        A = []
    return cb


