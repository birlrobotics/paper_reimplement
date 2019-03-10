import numpy as np
from dbscan_bhatt import DBSCAN 
from dbscan_bhatt import bhat_distance
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
X3 = np.array([[1, 2, 1], [2, 2, 2], [2, 3, 3], [8, 7, 7], [8, 8, 8], [25, 80, 50]])
clustering = DBSCAN(X, 2.1, 2)
# distances = bhat_distance(X[5], X[6])
print clustering