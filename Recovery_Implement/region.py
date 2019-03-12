# finished half, cluster, DBSCAN, B_distance should be done
# follow the data type of learn_state region function
from copy import copy
#  Arguments:
#       experience_list is a list of namedtuples: [(s,z,a,r,s',z'),(s,z,a,r,s',z'),...]
#       action_list is list of action space, generally from demo_record
#  Return:
#       a set of regions , each region is a set of bunch of experience tuples.
class Region_Cluster():

    def __init__(self):

        self.e_list = []
        self.a_dict = {}
        self.region_phi_result_set = set()

# Refer to Algorithm 1: Learning State Distribution Components
# phi_set is the lower case phi, region_phi_set is the capital case phi(which always has the prefix region_).
# Psi_set is also this case
"""Cluter a bunch of input points(states).

Parameters
----------
experience_list:(list)
    a list of experience namedtuples.

action_dict:(action_dict)
    a dict of demonstration actions.
Return
------
return_set : (set)
	a set of regions, the result of clusterring and merging.
    (psi_set, psi_set, ...)
    psi_set = set((s,z,a,r,s',z'),...)
"""
    def learn_state_region(self,experience_list,action_dict):
        self.e_list = experience_list
        self.a_dict = action_dict
        # line 2
        for act_index in self.a_dict.keys():
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
# distance_type: The distance type that DBSACNE uses to cluster.
#                      If it is 'states_dist', return a set of clusters , each cluster is a set of bunch of experience tuples.
#                      If it is 'regions_dist',  return a set of clusters , each cluster is a set of bunch of regions, and
#                      each region is a set of bunch of experience tuples.
# component: cluter on which component, 'states' or 'next_states'.

#  Return:
#       a set of clusters , each cluster is either a set of bunch of experience tuples,
#       or a set of rigion.

    def cluster(self, input_set, distance_type, component):
        TODO

 # DBSCAN algorithm

    def DBSCAN(self, distance_type):
        TODO


# Two kind of distance algorithms

# Ponit cluster distance
    def states_distance(self):
        TODO
# Region cluster distance
    def region_distance(self):
        TODO
