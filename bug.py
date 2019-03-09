import math
import numpy as np

def bhat_distance(a, b):
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    score = -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))
    return score

def get_neibor(data, dataset, e):
    res = []
    for i in range(np.shape(dataset)[0]):
        if bhat_distance(data , dataset[i])<e:
            res.append(i)
    return res

def DBSCAN(dataset, e, minpts):
    coreobjs = {}
    c = {}
    n = np.shape(dataset)[0]
    for i in range(n):