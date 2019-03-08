def dbscan(eps, minpts, samples):
	"""Define a clustering method use DBSCAN
	
	Parameters
	----------
	eps : float
		The maximum distance between two samples for \
		them to be considered as in the same neighborhood.
	minpte  : int
		The number of samples (or total weight) in a \
		neighborhood for a point to be considered as \
		a core point. This includes the point itself.
	samples : array_like
		The clustering sample states 
	Return
	------
	cluster.labels_ : array_like
		The clustered group of each data.
	"""
	from sklearn.cluster import DBSCAN
	cluster = DBSCAN(eps, minpts).fit(samples)
	return cluster.labels_

