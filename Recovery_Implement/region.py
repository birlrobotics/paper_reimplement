# finished half, cluster, DBSCAN, B_distance should be done
# follow the data type of learn_state region function

#  Arguments:
#       experience_list is a list of namedtuples: [(s,z,a,r,s',z'),(s,z,a,r,s',z'),...]
#       action_list is list of action space, generally from demo_record
#  Return:
#       a set of regions , each region is a set of bunch of experience tuples.
class Region_Cluster(experience_list,action_list):

    def __init__(self):

        self.e_list = experience_list
        self.a_list = action_list
        self.region_phy_result_set = set()
#       Refer to Algorithm 1: Learning State Distribution Components
# phy_set is the lower case phy, region_phy_set is the capital case phy(which always has the prefix region_).
    def learn_state_region(self):
        merged_region_phy_set = set()
        for act in self.a_list:
            rigon_phy_hat_set = set()
            same_act_exp_set = self.extract_exp_with_same_component('action',act)
            region_psi_set = self.cluster( same_act_exp_set ,distance_type = 'states_dist',component='states')
            # convert set to list for iteration
            list_region_psi_set = list(region_psi_set)
            # psi_set is a set of experience namedtuples within one output region psi,
            for psi_set in list_region_psi_set:
                region_phy_hat_hat_set = self.cluster(psi_set,distance_type = 'states_dist',component='next_states')
                # Union set operation
                region_phy_hat_set |= region_phy_hat_hat_set
            # P is a set of capital regions
            P = self.cluster(region_phy_hat_set,distance_type = 'states_dist',component='next_states')
            # Merge those phy_set in same region_phy_set
            for P_region_phy_set in P:
                merged_phy_set = set()
                for each_phy_set in P_region_phy_set:
                    merged_phy_set |= each_phy_set
                merged_region_phy_set.add(merged_phy_set)

            self.region_phy_result_set |= merged_region_phy_set

        return_set = self.region_phy_result_set
        return  return_set

#     Get a set of experience namedtuple with same component: [(s,z,a,r,s',z'),...]
    def extract_exp_with_same_component(self,type,component):
        if type == 'action':
            same_component_exp_set = set()
            for e in self.e_list:
                if e.a == component:
                    same_component_exp_set.add(e)

        elif type == 'next_state':
            same_component_exp_set = set()
            for e in self.e_list:
                if e.next_s == component:
                    same_component_exp_set.add(e)

        return same_component_exp_set


#  Arguments:
#       input_set: the set waited for cluster
#       input_exp_tuple_set: The distance type that DBSACNE uses to cluster.
#                       If it is 'states_dist', return a set of clusters , each cluster is a set of bunch of experience tuples.
#                       If it is 'regions_dist',  return a set of clusters , each cluster is a set of bunch of regions, and
#                         each region is a set of bunch of experience tuples.
#       component: cluter on which component, 'states' or 'next_states'.
#  Return:
#       a set of clusters , each cluster is either a set of bunch of experience tuples,
#       or a set of rigion.

    def cluster(self, input_exp_tuple_set, distance_type, component):
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
