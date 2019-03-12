import torch
import numpy as np
import gaussian_tool as gt
class Value_Function(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self):
         return
    def init_region_number(self,region_number, act_dict):
        self.region_q_array = np.ndarray.random.shape(region_number)
        self.demo_act_dict = act_dict
    """Q approximation function

    Parameters
    ----------
    state : (ndarray)
    	The position of agent. (2D or 3D)
    contact : (ndarray)
    	The contact mode of agent. (2D or 3D)
    regions_infs_list: (list)
        A list of regions informations: [phi_1,phi_2,...],
        phi_i = [phi_set, region_action, is_goal, mean, std, father, son]

    Return
    ---------
    s_a_q_arrary:(np.ndarray)
        1 dimension array recording each action value of a specific array.
    """
    def forward(self,state, contact, regions_infs_list):
        s_a_q_arrary = []
        for demo_action in self.demo_act_dict:
            value = 0
            numerator = 0
            denominator = 0
            for i, region_infs_list in enumerate(regions_infs_list):
                for region_action in region_infs_list[1]:
                    if demo_action == region_action:
                        phi_state_pro = gt.gaussian_prob(region_infs_list[3],region_infs_list[4],state, contact)
                        numerator += self.region_q_array[i] * phi_state_pro
                        denominator += phi_state_pro
                        value = numerator/denominator
                    else:
                        value = -inf
        s_a_q_arrary.append(value)
        return s_a_q_arrary
