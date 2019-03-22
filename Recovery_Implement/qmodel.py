import torch
import numpy as np
from scipy.stats import multivariate_normal

class Value_Function():
    def __init__(self):
        return
    def init_region_number(self,region_number, act_dict, phi_inf_list):
        
        self.region_q_array = torch.randn((region_number))
        self.region_q_array.requires_grad = True
        self.demo_act_dict = act_dict
        self.phi_inf_list = phi_inf_list

    def forward(self,state, contact):
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
        region_action = {a_index: goal, ...}
        Return
        ---------
        s_a_q_arrary:(np.ndarray)
        1 dimension array recording each action value of a specific array.
        """
        s_a_q_arrary = []
        for demo_action in self.demo_act_dict:
            value = 0
            numerator = 0
            denominator = 0
            for i, region_infs_list in enumerate(regions_infs_list):
                for region_action in region_infs_list[1]:
                    if demo_action == region_action:
                        phi_state_pro = multivariate_normal.pdf(state, contact, region_infs_list[3],region_infs_list[4])
                        numerator += self.region_q_array[i] * phi_state_pro
                        denominator += phi_state_pro
                        value = numerator/denominator
                    else:
                        value = -2000
            s_a_q_arrary.append(value)
        return s_a_q_arrary


    def test_forward(self,state):
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
        region_action = {a_index: goal, ...}
        Return
        ---------
        s_a_q_arrary:(np.ndarray)
        1 dimension array recording each action value of a specific array.
        """
        s_a_q_arrary = []
        for demo_action in self.demo_act_dict:
            value = 0
            numerator = 0
            denominator = 0
            for i, region_infs_list in enumerate(self.phi_inf_list):
                for region_action in region_infs_list[1]:
                    if demo_action == region_action:
                        phi_state_pro = multivariate_normal.pdf(state, region_infs_list[3],region_infs_list[4])
                        numerator += self.region_q_array[i] * phi_state_pro
                        denominator += phi_state_pro
                        value = numerator/denominator
                    else:
                        value = -2000
            s_a_q_arrary.append(value)
        s_a_q_arrary = np.array(s_a_q_arrary)
        return s_a_q_arrary



    def optim(self):
        self.region_q_array.data = self.region_q_array.data - 0.01 * self.region_q_array.grad.data 


    def zero_grad(self):
        self.region_q_array.grad.data.zero_()

    def get_region_q(self):
        print(self.region_q_array.data)