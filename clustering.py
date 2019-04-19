# -*- coding: utf-8 -*-
# Author: Jim Huang <huangjiancong863@gmail.com>
#
#
# License: MIT License

import math
import numpy as np
import random

UNCLASSIFIED = False
NOISE = -1 # the nosiy point is -1 in classification_set

class DBSCAN():
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise
            use Euclidean Distance D_s(s_i, s_j).
            We can switch the distance with Euclidean Distance or
            Bhattacharyya Distance by metric
    """

    def __init__(self, samples, eps, minpts, metric):
        self.samples = samples
        self.eps = eps
        self.minpts = minpts
        self.metric = metric

    def dbscan(self,):
        """Density-Based Spatial Clustering of Applications with Noise algo
        Parameters
        ----------
        samples : list
            The original samples waiting the clustering.
        eps : float
            The maximum distance between two samples for them to be considered \
            as in the same neighborhood. In the paper, the eps = 2cm.
        minpts : int
            The number of samples (or total weight) in a neighborhood for a \
            point to be considered as
            a core point. This includes the point itself.
        metric : str
            'E' or 'B'
            Use "Euclidean Distance" or "Bhattacharyya Distance".
        cluster_id : int
            The first cluster's number, for example No.1 cluster
        Return
        ------
        clusters : array_like
            The clustered group of each data.
        """
        cluster_id = 1
        n_points = len(self.samples)
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            # point = samples[:,point_id]
            if classifications[point_id] == UNCLASSIFIED:
                if self.expand_cluster(classifications, point_id, cluster_id):
                    cluster_id = cluster_id + 1
        return classifications

    def expand_cluster(self, classifications, point_id, cluster_id):
        seeds = self.region_query(point_id)
        if len(seeds) < self.minpts:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id
            while len(seeds) > 0:
                current_point = seeds[0]
                results = self.region_query(current_point)
                if len(results) >= self.minpts:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or \
                        classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
        return True

    def region_query(self, point_id):
        n_points = len(self.samples)
        seeds = []
        for i in range(0, n_points):
            if self.eps_neighborhood(self.samples[point_id], self.samples[i]):
                seeds.append(i)
        return seeds

    def eps_neighborhood(self, point_a, point_b):
        if self.metric == 'E':
            # Caculate the Euclidean Distance with state and contact mode at \
            #   the same time
            # Like: D_s(s_i; s_j) = ||s_i − s_j||_2 * ||z^m_i − z^m_j||_2
            return self.eucli_distance(point_a[0], point_b[0]) < self.eps
        else:
            # Caculate the bhatt distance between two distribution
            return self.bhat_distance(point_a, point_b) < self.eps

    def bhat_distance(self, obs_a, obs_b):
        """Caculate the Bhattacharyya Distance of every oberservations in point
        Reference
        ---------
        https://en.wikipedia.org/wiki/Bhattacharyya_distance
        """
        if not len(obs_a) == len(obs_b):
            raise ValueError("a and b must be of the same size")
        DIFF = np.array(obs_a[0])-np.array(obs_b[0])
        SUM = (np.array(obs_a[1])+np.array(obs_b[1]))*0.5
        DET = np.linalg.det(SUM)/np.sqrt(np.linalg.det(np.array(obs_a[1]))* \
        np.linalg.det(np.array(obs_b[1])))

        return 0.125*np.dot(np.dot(DIFF, np.linalg.inv(SUM)), DIFF.T) + \
         0.5*np.log10(DET)

    def eucli_distance(self, obs_a, obs_b):
        """Caculate the Euclidean Distance of every oberservations in point"""
        if not len(obs_a) == len(obs_b) and len():
            raise ValueError("a and b must be of the same size")
        return np.linalg.norm(np.array(obs_a) - np.array(obs_b))