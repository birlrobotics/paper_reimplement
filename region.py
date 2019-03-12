# -*- coding: utf-8 -*-
"""
Cluter a bunch of input points(states)
"""

# Author: Jim Huang <huangjiancong863@gmail.com>
#         Bourne Lin <yijionglin.bourne@gmail.com>
#
# License: MIT License


from dbscan import DBSCAN


class Region_Cluster():
    """Learning stribution State Distribution Components.

    Parameters
    ----------
    experience_list : list
        a list of experience namedtuples: [(s,z,a,r,s',z'),(s,z,a,r,s',z'),...].
    action_list : 
        list of action space, generally from demo_record
    action_dict : 
        a dict of demonstration actions.

    Return
    ------
    return_set : 
	    a set of regions, the result of clusterring and merging.(psi_set, psi_set, ...)
        psi_set = set((s,z,a,r,s',z'),...)
    
    Examples
    --------
    >>>

    Notes
    -----
    phi_set is the lower case phi, region_phi_set is the capital case phi(which always has the prefix region_).
    psi_set is the lower case psi, region_psi_set is the capital case psi(which always has the prefix region_).

    References
    ----------

    """

    def __init__(self, experience_list, action_dict):
        self.e_list = experience_list
        self.a_dict = action_dict
        self.region_phi_result_set = set()

    def learn_state_region(self, ):
        """

        """
        
        # line 2
        for act_index in self.a_dict:
            # line 3

            region_phi_hat_set = set()

            same_act_exp_set = self.extract_exp_with_same_action(act_index)
            region_psi_set = self.cluster(same_act_exp_set ,distance_type = 'states_dist',component='next_states')
            # convert set to list for iteration
            list_region_psi_set = list(region_psi_set)
            # psi_set is a set of experience namedtuples within one output region psi,
            for psi_set in list_region_psi_set:
                region_phi_hat_hat_set = self.cluster(psi_set,distance_type = 'states_dist',component='states')
                # Union set operation
                region_phi_hat_set |= region_phi_hat_hat_set
            # Line 11-12
            # P is a set of capital regions
            P = self.cluster(region_phi_hat_set,distance_type = 'region_dist', component= null)
            # Merge those phi_set in same region_phi_set
            merged_region_phi_set = set()
            for P_region_phi_set in P:
                merged_phi_set = set()
                for each_phi_set in P_region_phi_set:
                    merged_phi_set |= each_phi_set
                merged_region_phi_set.add(merged_phi_set)
            self.region_phi_result_set |= merged_region_phi_set

        return_set = self.region_phi_result_set
        return  return_set

#     Get a set of experience namedtuple with same action: [(s,z,a,r,s',z'),...]
    def extract_exp_with_same_action(self,act_index):
        same_component_exp_set = set()
        for e in self.e_list:
            # each exp tuple just have one action, so donnot need to iterate, just take the first one
            action_list = list(e.action.keys())
            if action_list[0] == act_index:
                same_component_exp_set.add(e)
        return same_component_exp_set


# Args:
# input_set: the set waited for cluster
# distance_type: 
# component: .

#  Return:
#       

    def cluster(self, input_set, eps, minpts, distance_type, component):
        """Create a clustered brunch of points
        Parameters
        ----------
        input_set : array_like
            The original samples waiting the clustering.
        eps : float
		    The maximum distance between two samples for them to be considered as in
		    the same neighborhood. In the paper, the eps = 2cm.
	    minpts : int
		    The number of samples (or total weight) in a neighborhood for a point to
		    be considered as a core point. This includes the point itself.
        distance_type : str
            The distance type that DBSACNE uses to cluster.
            If it is 'states_dist', return a set of clusters, \
            each cluster is a set of bunch of experience tuples.
            If it is 'regions_dist', return a set of clusters, \
            each cluster is a set of bunch of regions.
            And each region is a set of bunch of experience tuples.
        component : str
            Cluter on which component, 'states' or 'next_states'.

        Return
        ------
        cb : list
            a set of clusters , each cluster is either a set of bunch of experience tuples,
            or a set of region.
        """
        # use DBSCAN with Euclidean Distance to create the states bunch
        if distance_type == 'states_dist':
            minpts = 3
            classifications = DBSCAN(input_set, 2, minpts, metric='E')
            cb = []
            A = []
            # search the index with same value
            for i in set(classifications):
                address_index = [x for x in range(len(classifications)) if classifications[x] == i]
                for j in address_index:
                    A.append(input_set[j])
                    cb.append(A)
                    A = []

        # use DBSCAN with Bhattacharyya Distance to create the distribution bunch
        else:
            minpts = 3
            classifications = DBSCAN(input_set, 0.5, minpts, metric='B')
            cb = []
            A = []
            # search the index with same value
            for i in set(classifications):
                address_index = [x for x in range(len(classifications)) if classifications[x] == i]
                for j in address_index:
                    A.append(input_set[j])
                    cb.append(A)
                    A = []
        return cb






# Two kind of distance algorithms

# Ponit cluster distance
    def states_distance(self):
        TODO
# Region cluster distance
    def region_distance(self):
        TODO
