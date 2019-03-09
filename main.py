import numpy as np
from sklearn.cluster import DBSCAN
import ipdb; ipdb.set_trace()
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
clustering2 = DBSCAN(eps=1, min_samples=3).fit(X)
