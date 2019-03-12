import numpy as np
from dbscan import DBSCAN 
from dbscan import bhat_distance
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
X3 = np.array([[1, 2, 1], [2, 2, 2], [2, 3, 3], [8, 7, 7], [8, 8, 8], [25, 80, 50], [26, 82, 52]])


# clustering = DBSCAN(X3, 2, 2, metric='E')
# # distances = bhat_distance(X[5], X[6])
# print clustering
# print type(clustering)

# from clustering import clustered_batch    
# clustered_samples = clustered_batch(clustering, X3)
# # dict_address = dict(clustered_samples)
# print clustered_samples
# print type(clustered_samples)
# print len(clustered_samples)


r = np.cov(X3.T)
eigen_vals, eigen_vect = np.linalg.eig(r)

cov_matrix = np.diag(eigen_vals) # cov
mean_m = np.mean(X3, axis=0) # mean
# print mean_m
# print cov_matrix
phi_s = np.random.multivariate_normal(mean_m, cov_matrix, (1, 2, 3))
# print phi_s

from scipy.stats import multivariate_normal
psi_s = multivariate_normal((1, 2, 3), mean_m, cov_matrix)
print psi_s
