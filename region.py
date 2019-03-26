# -*- coding: utf-8 -*-
"""
Cluter a bunch of input points(states)
"""

# Author: Jim Huang <huangjiancong863@gmail.com>
#         Bourne Lin <yijionglin.bourne@gmail.com>
#
# License: MIT License


import numpy as np
import clustering



class Region_Cluster():
    """Learning stribution State Distribution Components.

    Parameters
    ----------
    experience_list : list
        a list of experience namedtuples: [(s,z,a,r,s',z'),(s,z,a,r,s',z'),...].
    action_dict : list
        a dict of demonstration actions.

    Return
    ------
    return_set : list
	    A set of each region which having mean and covariance matrix correspond.

    Examples
    --------
    >>>

    Notes
    -----
    phi_set is the lower case phi, region_phi_set is the capital case phi\
        (which always has the prefix region_).
    psi_set is the lower case psi, region_psi_set is the capital case psi\
        (which always has the prefix region_).

    References
    ----------

    """


    def __init__(self, experience_list, action_dict):
        self.e_list = experience_list
        self.a_dict = action_dict

    def learn_state_region(self,):
        """
        Algo 1 in the recovery paper and add total DBSCAN with bhatt distance \
         in the end
        """
        # line 2
        return_set_set = []
        for act_index in self.a_dict:
            # line 3
            region_phi_hat_set = []
            same_act_exp_set = self.extract_exp_with_same_action(act_index)
            region_psi_set = self.cluster(same_act_exp_set, component='next', \
                distance_type = 'states_dist')
            # convert set to list for iteration
            list_region_psi_set = list(region_psi_set)
            # psi_set is a set of experience_list within one output region psi,
            for psi_set in list_region_psi_set:
                region_phi_hat_hat_set = self.cluster(psi_set, \
                    component='current', distance_type = 'states_dist')
                # Union set operation
                region_phi_hat_set.extend(region_phi_hat_hat_set)
            # Line 11-12
            # P is a set of capital regions
            P = self.cluster(region_phi_hat_set, component= 'current', \
                distance_type = 'region_dist')
            if P != []:
                return_set_set.extend(P)
        #At the end, use Bhatt distance to cluster with total region after \
        # clustered by every same action set
        ## TODO:check bhatt distance
        cluster = clustering.DBSCAN(return_set_set, 0.05, minpts=2, metric='B')
        classifications_b = cluster.dbscan()
        # The six lines below are to save the single distribution region
        # Becasue there have no noise region in this region_set
        if classifications_b != []:
            a = max(classifications_b)
            address_class = [x for x in range(len(classifications_b)) if\
                classifications_b[x] == -1]
            for k in address_class:
                classifications_b[k]=1+a
                a += 1
        return_set = self.clustered_batch(classifications_b, return_set_set, \
            metric='extend')
        return return_set

    def extract_exp_with_same_action(self, act_index):
        """Get a set of experience namedtuple with same action:\
         [(s,z,a,r,s',z'),...]
        """
        same_component_exp_set = []
        for e in self.e_list:
            # each exp tuple just have one action, so donnot need to iterate, \
            # just take the first one
            action_list = list(e.action.keys())
            if action_list[0] == act_index:
                same_component_exp_set.append(e)
        return same_component_exp_set

    def cluster(self, input_set, component, distance_type):
        """Create a clustered brunch of points
        Parameters
        ----------
        input_set : array_like
            The original samples waiting the clustering.
        eps : float
		    The maximum distance between two samples for them to be considered \
            as in the same neighborhood. In the paper, the eps = 2cm.
            Like a threshold.
	    minpts : int
		    The number of samples (or total weight) in a neighborhood for a \
            point to be considered as a core point.
            This includes the point itself.
        distance_type : str
            The distance type that DBSACNE uses to cluster.
            If it is 'states_dist', return a set of clusters, each cluster is \
            a set of bunch of experience tuples.
            If it is 'regions_dist', return a set of clusters, each cluster is \
            a set of bunch of regions.
            And each region is a set of bunch of experience tuples.
        component : str
            Cluster on which component, 'states' or 'next_states'.

        Return
        ------
        cb : list
            A set of clusters , each cluster is either a set of bunch of \
            experience tuples, or a set of region. where each subset is a set \
            containing all states or region belonging to that cluster
        """

        if component == 'current':
            if distance_type == 'states_dist':
                # use DBSCAN with Euclidean Distance to create the states bunch
                train_set_cs = self.extract_sz(input_set, component)
                cluster = clustering.DBSCAN(train_set_cs, eps=5, minpts=3, \
                    metric='E')
                classifications_cs = cluster.dbscan()
                cb = self.clustered_batch(classifications_cs, train_set_cs, \
                    metric='append')
                return cb
            else:
                # use DBSCAN with Bhat Distance to create the distribution bunch
                d = self.moment(input_set) # return the distribution set
                cluster = clustering.DBSCAN(d, 0.05, minpts=2, metric='B')## TODO: verify with 1-D 2_D
                classifications_b = cluster.dbscan()
                # The six lines below are to save the single distribution region
                # Becasue there have no noise region in this region_set
                if classifications_b != []:
                    a = max(classifications_b)
                    address_class = [x for x in range(len(classifications_b)) if\
                        classifications_b[x] == -1]
                    for k in address_class:
                        classifications_b[k]=1+a
                        a += 1
                cb = self.clustered_batch(classifications_b, d, metric='extend')
                return cb

        else:
            train_set_ns = self.extract_sz(input_set, component)
            cluster = clustering.DBSCAN(train_set_ns, eps=5, minpts=3, \
                # TODO:eps threshold
                metric='E')# ros use standard unit(kg, metres, radians)
            classifications_ns = cluster.dbscan()
            cb = self.clustered_batch(classifications_ns, input_set, \
                metric='append')
            return cb

    def clustered_batch(self, classifications, samples, metric):
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
        SET = list(set(classifications))
        if -1 in SET:
            SET.remove(-1) # remove the noise point
        for i in SET:
            address_index = [x for x in range(len(classifications)) if \
                classifications[x] == i]
            for j in address_index:
                if metric == 'append':
                    A.append(samples[j])
                else:
                    A.extend(samples[j])
            cb.append(A)
            A = []
        return cb

    def moment(self, batch):
        """
        Cluster the moment(distribution) use Bhattacharyya Distances。
        In the paper, the max distance eps_psi or eps_phi is 0.05cm
        Step 1. Use Maximum Likelihood to find the mean and covariance matrix
        Step 2. Use multivariate normal distribution to compute the distribution
        Parameters
        ----------
        batch : array_like
            The clustered samples' batchs.

        Return
        ------
        cb : array_like
            Having mean and covariance matrix

        """
        cb_b = []
        for i in range(len(batch)):
            samples = batch[i]
            concatenate = []
            for sample in samples:
                concatenate.append(np.append(sample[0],sample[1]))
            concatenate = np.array(concatenate)
            me = np.mean(concatenate, axis=0) # the mean of every dimension
            co = np.cov(concatenate.T) # the covariance matrix of samples
            # the distribution of point s in cluster
            if np.linalg.det(co) != 0:
                cb_b.append([me, co])
        cb = cb_b
        return cb

    def extract_sz(self, input_set, component):
        """extract the state and contact mode with same action"""
        train_set = []
        for e in input_set:
            if component == 'current':
                train_set.append([e.state, e.contact])
            else:
                train_set.append([e.next_state, e.next_contact])
        return train_set
