# -*- coding: utf-8 -*-
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise 
        use Bhattacharyya Distance.
"""

# Author: Jim Huang <huangjiancong863@gmail.com>
#         
#
# License: MIT

import math
import numpy as np
import random

def bhat_distance(a, b):
    """Caculate to Bhattacharyya Distance"""
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    score = -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))
    return score

def get_neibor(data, dataset, e):
    """Get the ε-Area"""
    res = []
    for i in range(np.shape(dataset)[0]):
        if bhat_distance(data , dataset[i])<e:
            res.append(i)
    return res

def DBSCAN(dataset, eps, minpts):
    """Density-Based Spatial Clustering of Applications with Noise algo 
       Caculate to Bhattacharyya Distance
    Parameters
	----------
	eps : float
		The maximum distance between two samples for them to be considered as in
		the same neighborhood.
	minpts : int
		The number of samples (or total weight) in a neighborhood for a point to
		be considered as
		a core point. This includes the point itself.
	dataset : array_like
		The clustering sample states.
	Return
	------
	c : array_like
		The clustered group of each data.
	"""
    core_objs = [] # Initialize the core object(index)
    c = []
    n = np.shape(dataset)[0]
    # Find all core objects, the key is the index of the core object, and the \
	# value is the inde of the object in the ε-neighborhood.
    
    for i in range(n):
        neibor = get_neibor(dataset[i], dataset, eps)
        if len(neibor)>=minpts:
            core_objs[i] = neibor
    old_core_objs = core_objs.copy()
    k = 0
    not_access = range(n)
    while len(core_objs)>0:
        old_not_access = []
        old_not_access.extend(not_access)
        cores = core_objs.keys()
        # Randomly select a core object
        rand_num = random.randint(0, len(cores))
        core = cores[rand_num]
        queue = []
        queue.append(core)
        not_access.remove(core)
        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in old_core_objs.keys():
                delte = [val for val in old_core_objs[q] if val in not_access] #Δ = N(q)∩Γ
                queue.extend(delte) # Add the sample in Δ to the queue Q
                not_access = [val for val in not_access if val not in delte] #Γ = Γ\Δ
        k += 1
        c[k] = [val for val in old_core_objs if val not in not_access]
        for x in c[k]:
            if x in core_objs.keys():
                del core_objs[x]
    return c