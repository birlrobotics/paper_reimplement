import numpy as np
from test import DBSCAN
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
m = np.matrix('1 1.2 0.8 3.7 3.9 3.6 10; 1.1 0.8 1 4 3.9 4.1 10')
clustering = DBSCAN(m, 0.5, 2)
print clustering