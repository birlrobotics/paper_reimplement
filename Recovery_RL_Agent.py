"""descibision of a function

Parameters
----------
arg1 : ((type))
    descibision
arg2 : (type)
    descibision

Return
------
arg1 : (type)
    descibision
"""
"""
"""
from copy import deepcopy
import random
import numpy as np
# import gaussian_tool as gt
from buffer import Exp_Buffer
from region import Region_Cluster
from collections import OrderedDict
from qmodel import Value_Function

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 5e-4               # learning rate 

class Agent():

    def __init__(self, seed = 0):

        """
        self.demo_act = {a_i: goal_postion}
        """
        self.demo_act_dict = OrderedDict()
        self.num_of_demo_goal = OrderedDict()
        self.state_value_func = Value_Function()
        self.exps_list = Exp_Buffer()
        self.cluster = Region_Cluster(experience_list=[],action_dict={})

        """
        self.act for new skills
        """
        
        self.actions_dict = OrderedDict()
        self.state_size = 0
        self.action_size = 0
        self.seed = random.seed(seed)


        """
        self.region_dict:(dictionary)
        {region_index: phi_set},
        region_index = i
        phi_set = set((s,z,a,r,next_s,next_z),...)
        """
        self.regions_infs_list=[]
        self.region_dict={}
        self.num_of_region = 0


    def demo_record(self,goal_tuples):
        """To record a task demonstration skill with several goal states,
        i.e. the goal 3D position of each initial skill.
        Use ROS interface to get the position information of robot.
        Parameters
        ----------
        goal_tuples:(2 dim array)
        """
        for i, goal in enumerate(goal_tuples):
            action_index = str(i)
            self.demo_act_dict[action_index] = goal
        demo_act_dict = self.demo_act_dict
        return demo_act_dict

    def get_demo_act_dict(self):
        """Return the action dictionary of demonstration
        Return
        ----------
        demo_act_dict:(dict)
            keys: index string
            values: goal position
        """
        demo_act_dict = self.demo_act_dict
        return demo_act_dict

    def exp_record(self,episode_list):
        """Record experience tuples of an episode to experience list
        Param
        ----------
        episode_list:(list)
            a list of experience tuple.
        """
        for exp_tuple in episode_list:
            state, contact, action, reward, next_state, next_contact, is_goal = exp_tuple
            self.exps_list.add(state, contact, action, reward, next_state, next_contact, is_goal)

    # return a list of namedtuple for experience.
    def get_exp_list(self):
        """Return the list of all the restored experience tuples
        Return
        ----------
        experiences_list:(list)
            a list of experience tuple.
        """
        experiences_list = self.exps_list.get_experience_list()
        return experiences_list



    def learn_cluster_region(self):
        """Call the cluster method to get regions. And assign the relative action to each region.
        Also determine a region whether it is a goal region.

        region_phi_result_set:(set)
            a set of regions, the result of clusterring and merging.
            (psi_set, psi_set, ...)
            psi_set = set((s,z,a,r,s',z'),...)

        regions_infs_list = [phi_1, phi_2, ...]
            phi_i = [phi_set, act_dict]
            phi_set = ((state,contact,action,reward,next_state,next_contact),...)
            act_dict = [a_index: goal_postion]
        """
        pass
        exp_tuples_list = self.get_exp_list()
        region_phi_result_set = self.cluster.learn_state_region(exp_tuples_list, self.demo_act_dict)
        for i, phi_set in enumerate(region_phi_result_set):
            phi_list = phi_set
            self.regions_infs_list.append([])
            self.regions_infs_list[i].append(deepcopy(phi_set))
            # phi_list.[0].a is  {a_index:goal_position}
            self.regions_infs_list[i].append(deepcopy(phi_list[0].action))
            # determine a region if it is goal region, by checking its action whether is empty or not.
            if bool(phi_list[0].action) == False:
                self.regions_infs_list[i].append(True)
            else:
                self.regions_infs_list[i].append(False)

        self.num_of_region = len(region_phi_result_set)


    def get_region_number(self):
        """Get the number of regions.
        Return
        ------
        region_number:(int)
        """
        pass
        region_number = self.num_of_region
        return region_number


    def gaussian_likelihood(self):
        """Call the gaussian maximum likelihood estimation method.
        Put mean and std in each region_infs_list's phi_i 3rd and 4th index of place.
        regions_infs_list = [phi_1, phi_2, ...]
            phi_i = [phi_set, act_dict] ->  phi_i = [phi_set, act_dict, mean, std]
                phi_set = ((state,contact,action,reward,next_state,next_contact),...)
                act_dict = [a_index: goal_postion]
                mean = array()
                std = array()
        """
        pass
        for i, region_infs_list in enumerate(self.regions_infs_list):
            mean, std = gt.gaussian_estimation(region_infs_list[0])
            self.regions_infs_list[i].append(mean)
            self.regions_infs_list[i].append(std)



    def region_directed_graph(self):
        """Construct region directed graph used for algorithm 2.
        """
        pass

    def test_init_value_function(self,phi_inf_list):
        """Initialization of the approximate value function with the number of region.
        """
        self.state_value_func.init_region_number(3, self.demo_act_dict, phi_inf_list)

    def test_learn_initial_policy(self):
        """Learn the policy with the experience tuple.
        """
        
        for e in self.exps_list.get_experience_list():
            state, contact, action, reward, next_state, next_contact, done = e
            Q_target_next = self.state_value_func.test_forward(next_state)
            Q_target = reward + Q_target_next *(1 - done)

#           Get the action index
            a = int(action.keys()[0])
            Q_expected = self.state_value_func.test_forward(state)[a]
                        
            loss = F.mse_loss(Q_expected, Q_target)
            
            self.state_value_func.optim()
            self.state_value_func.zero_grad()

        #  not done
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        pass
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def choose_act(self, state, action, recovery_prob = 0):
        """Call the gaussian maximum likelihood estimation method.

        Return
        ------
        self.region_prob_list:(list)
        [region_prob_tuple], region_prob_tuple = (region_index,mean,std,is_goal)
        """
        pass
    def learn_Q_value(self, ):
        """Update Q function with experience tuple
        """
        pass
