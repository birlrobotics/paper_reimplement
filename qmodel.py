from __future__ import print_function
import torch
import numpy as np
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from model import LR

class Value_Function():
    def __init__(self, lr, gamma, tau, init_Q_value, if_soft_update, funnels_inf_list, demo_acts_dict):
        """
        Param:
            lr(float):learning rate
            gamme(float):discount factor
            tau(float):the parameter of soft update
            init_Q_value(float): the initializaed Q table value and q value. Be careful if the init value is bigger than true Q value, there will be a problem in convergence.
            if_soft_update(bool): Use soft update if true.
            acts_dict(OrderedDict): Dictionaries of all the actions. {'a_i':goal_position}
            funnels_inf_list(list): a list of funnels informations. Each element follows this form: [funnel_index(str),act_dict(dict),region_dict(dict)]
                                    which act_dict = {'a_i':goal_pos}region_dict= {'r_i':(mean, cov)}
            
        Note: 
            Three kinds of iteration: for all action, for all input batch states, for all funnels.
        """
        # Hyperparameters
        self.lr =lr
        self.gamma = gamma
        self.tau = tau
        self.init_Q_value = init_Q_value
        self.if_soft_update = if_soft_update
        # funnels inf
        self.funnels_inf_list = funnels_inf_list
        self.funnels_amount = len(funnels_inf_list)


# -----------------------------New approximation-----------------------------
        self.local_q_model = LR(demo_acts_dict, funnels_inf_list,init_Q_value)
        self.target_q_model = LR(demo_acts_dict, funnels_inf_list,init_Q_value)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.local_q_model.parameters(), lr=lr)

# ============================================Init Finished ============================================

# # -------------------------New approximation----------------------------------------------------------------------------
    
    def q_learn(self,experiences):
        """
        Update the q parameters with expereience tuples.

        Param:
        experiences(tuple): a tuple of states,actions,rewards,next_states,dones, and each of the tuple element is a tensor type.
        """
        states,actions,rewards,next_states,dones = experiences
        # forward
        if self.if_soft_update:
            Q_targets_next = self.target_q_model.qforward(next_states).detach().max(1)[0].unsqueeze(dim=1)
        else:
            Q_targets_next = self.local_q_model.qforward(next_states).max(1)[0].unsqueeze(dim=1)
        Q_targets = rewards + self.gamma*Q_targets_next * (1-dones)
        Q_expects = self.local_q_model.qforward(states).gather(1,actions)
        loss = self.mse_loss(Q_expects, Q_targets)

        # backward
        self.optimizer.zero_grad() # clear grad
        loss.backward()
        self.optimizer.step()
        return loss
    
    def soft_update(self):
        """
        Fixed target
        """
        for local_param, target_param in zip(self.local_q_model.parameters(), self.target_q_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def forward(self,state):
        Q_s_tensor = self.local_q_model.qforward(state).max(0)
        return Q_s_tensor

    def get_param(self):
        """
        Return the q.
        """
        for param in self.local_q_model.parameters():
            return(param)

# # -------------------------New approximation----------------------------------------------------------------------------