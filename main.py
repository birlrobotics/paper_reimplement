import numpy as np
from dbscan_bhatt import DBSCAN
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(X, 3, 2).fit(X)
