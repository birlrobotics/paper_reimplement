import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal

class LR(nn.Module):
    def __init__(self,acts_dict, funnels_inf_list,init_Q_value):
        """
        Param:
            acts_dict(OrderedDict): Dictionaries of all the actions. {'a_i':goal_position}
            funnels_inf_list(list): a list of funnels informations. Each element follows this form: [funnel_index(str),act_dict(dict),region_dict(dict)]
                                    which act_dict = {'a_i':goal_pos}region_dict= {'r_i':(mean, cov)}
            init_Q_value(float): the initializaed Q table value and q value. Be careful if the init value is bigger than true Q value, there will be a problem in convergence.
            
        Note: 
            Three kinds of iteration: for all action, for all input batch states, for all funnels.
        """
        super(LR, self).__init__()
        self.funnels_amount = len(funnels_inf_list)
        self.funnels_inf_list = funnels_inf_list
        self.acts_dict = acts_dict
        self.action_size = len(acts_dict)
        #the input of model is phi*delta, input size = funnels amount, output size = 1
        self.q_network = nn.Linear(self.funnels_amount, 1,bias = False) 
        self.init_Q_value = init_Q_value
        #  For restoring the q parameters and uploaded for changing the network.
        self.q_params_save = 0

    def qforward(self,states):
        """
        The region Q value function forward pass.

        Param:
            states(tensor): input a batch of states to compute the relative Q value.
        """
        # Q_s table size: batch_size * action_size
        Q_s = torch.ones(len(states),self.action_size) * self.init_Q_value
#         Iterate all action
        for i,(key,value) in enumerate(self.acts_dict.iteritems()):
            
            a_dict = {}
            a_dict[key] = value
            
            # mul_fac size: funnels_amount * 1
            for j,state in enumerate(states):
                mul_fac = torch.from_numpy(self.multiply_factor(state,a_dict)).unsqueeze(dim=0).float()
                numerator =self.q_network(mul_fac)
                denominator= sum(mul_fac.squeeze())
                # If the denominator is 0, there will be an NaN result
                if denominator  >= 1.0e-30:
                    Q_s_a = numerator/denominator
                else:
                    Q_s_a = self.init_Q_value
                Q_s[j][i] = Q_s_a
        return Q_s
    
    # funnel_infs_list[f_i, action{a_i: goal},region{r_i: mean,std}]
    # return: size funnels_amount * 1
    def multiply_factor(self,state,a_dict):
        """
        Initialize two vectors to restore the phi and delta value for all funnel in of input s,a
        Compute the multiply_factor, which is phi * delta

        Param:
            state(tensor):input one state, which relative to the Q(s,a)
            a_dict(dict): input one action dictionary, which relative to the Q(s,a)

        Return:
            mul_fac(np.array): size is (funnels_amount,1)   
        """ 
        if_act_same = np.zeros(self.funnels_amount)
        region_prob = np.zeros(self.funnels_amount)
        for i,a_i, r_i, i_son, i_father in self.funnels_inf_list:
            i = int(i)
    #              delta function for action. If the actions are same, set relative region to 1
            if a_i.keys() == a_dict.keys():
                if_act_same[i] = 1
            region_prob[i] = multivariate_normal.pdf(state, r_i.values()[0][0], r_i.values()[0][1])

    #             Compute all the region's distribution
        mul_fac = region_prob * if_act_same
        return mul_fac