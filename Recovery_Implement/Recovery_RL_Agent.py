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
# from region import Region_Cluster
from collections import OrderedDict
from qmodel import Value_Function

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 5e-4               # learning rate 
BATCH_SIZE = 3
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class Agent():

    def __init__(self, seed = 0, dim =2 ):

        """
        self.demo_act = {a_i: goal_postion}
        """
        self.demo_act_dict = OrderedDict()
        self.num_of_demo_goal = OrderedDict()
        self.demo_goal = []
        self.state_value_func = Value_Function()
        self.memory = Exp_Buffer(batch_size= BATCH_SIZE)

        # The dimension of the workspace (2 or 3).
        self.dim = dim
#         self.cluster = Region_Cluster()

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
        self.region_amount= 0
        self.seperate_s_c_exp_list = []

    def demo_record(self,goal_tuples):
        """To record a task demonstration skill with several goal states,
        i.e. the goal 3D position of each initial skill.
        Use ROS interface to get the position information of robot.
        Parameters
        ----------
        goal_tuples:(2 dim array)
        """
        self.demo_goal = goal_tuples
        self.demo_amount = len(goal_tuples)
        for i, goal in enumerate(goal_tuples):
            action_index = str(i)
            self.demo_act_dict[action_index] = goal
            

        print("Agent: agent has recorded the demonstration as its original skills: {} \n".format(self.demo_act_dict))
        return self.demo_act_dict

    def get_demo_act_dict(self):
        """Return the action dictionary of demonstration
        Return
        ----------
        demo_act_dict:(dict)
            keys: index string
            values: goal position
        """
        self.demo_act_dict
        return self.demo_act_dict

    def exp_record(self,episode_list):
        """Record experience tuples of an episode to experience list
        Param
        ----------
        episode_list:(list)
            a list of experience tuple.
        """
        for exp_tuple in episode_list:
            state, action, reward, next_state, is_goal = exp_tuple
            self.memory.add(state, action, reward, next_state, is_goal)
    
    def seperate_exp_record(self,episode_list):
        """Record experience tuples of an episode to experience list
        Param
        ----------
        episode_list:(list)
            a list of experience tuple.
        """
        for exp_tuple in episode_list:
            state,contact,  action, reward, next_state,  next_contact, done = exp_tuple
            self.memory.seperate_add( state,contact,  action, reward, next_state,  next_contact, done)

    # return a list of namedtuple for experience.
    def get_exp_list(self):
        """Return the list of all the restored experience tuples
        Return
        ----------
        experiences_list:(list)
            a list of experience tuple.
        """
        experiences_list = self.memory.get_experience_list()
        return experiences_list

    def get_seperate_exp_list(self):
        """Return the list of all the restored experience tuples
        Return
        ----------
        experiences_list:(list)
            a list of experience tuple.
        """
        return self.memory.get_seperate_exp_list()

    # def learn_cluster_region(self):
    #     """Call the cluster method to get regions. And assign the relative action to each region.
    #     Also determine a region whether it is a goal region.

    #     region_phi_result_set:(set)
    #         a set of regions, the result of clusterring and merging.
    #         (psi_set, psi_set, ...)
    #         psi_set = set((s,z,a,r,s',z'),...)

    #     regions_infs_list = [phi_1, phi_2, ...]
    #         phi_i = [phi_set, act_dict]
    #         phi_set = ((state,contact,action,reward,next_state,next_contact),...)
    #         act_dict = [a_index: goal_postion]
    #     """
    #     pass
    #     exp_tuples_list = self.get_exp_list()
    #     region_phi_result_set = self.cluster.learn_state_region(exp_tuples_list, self.demo_act_dict)
    #     for i, phi_set in enumerate(region_phi_result_set):
    #         phi_list = phi_set
    #         self.regions_infs_list.append([])
    #         self.regions_infs_list[i].append(deepcopy(phi_set))
    #         # phi_list.[0].a is  {a_index:goal_position}
    #         self.regions_infs_list[i].append(deepcopy(phi_list[0].action))
    #         # determine a region if it is goal region, by checking its action whether is empty or not.
    #         if bool(phi_list[0].action) == False:
    #             self.regions_infs_list[i].append(True)
    #         else:
    #             self.regions_infs_list[i].append(False)

    #     self.num_of_region = len(region_phi_result_set)


    def get_region_amount(self):
        """Get the amount of regions.
        Return
        ------
        self.region_amount(int):
        """
        pass
        return self.region_amount


    # def gaussian_likelihood(self):
    #     """Call the gaussian maximum likelihood estimation method.
    #     Put mean and std in each region_infs_list's phi_i 3rd and 4th index of place.
    #     regions_infs_list = [phi_1, phi_2, ...]
    #         phi_i = [phi_set, act_dict] ->  phi_i = [phi_set, act_dict, mean, std]
    #             phi_set = ((state,contact,action,reward,next_state,next_contact),...)
    #             act_dict = [a_index: goal_postion]
    #             mean = array()
    #             std = array()
    #     """
    #     pass
    #     for i, region_infs_list in enumerate(self.regions_infs_list):
    #         mean, std = gt.gaussian_estimation(region_infs_list[0])
    #         self.regions_infs_list[i].append(mean)
    #         self.regions_infs_list[i].append(std)



    def region_directed_graph(self):
        """Construct region directed graph used for algorithm 2.
        """
        pass


    def test_get_phi_inf_list(self):
        cov = np.eye(self.dim * 3)
        regions = []
        origional_point =  np.zeros(self.dim)
        regions.append(origional_point)
        for each_goal in self.demo_goal:
            regions.append(each_goal)

        self.region_amount = len(regions)

        regions_infs_list=[]
        for region_position, (key,value) in zip(regions,self.demo_act_dict.items()):
            alist = []
            act_dict = {}
            act_dict[key] = value
            alist.append(key)
            alist.append(act_dict)

            # Generate the mean covering the position and contact
            contact_mean = self.test_get_contact_mean(region_position)
            mean = np.append(region_position, contact_mean)

            alist.append(mean)
            alist.append(cov)
            regions_infs_list.append(alist)
        print("Agent: test region infs:{} \n".format(regions_infs_list))
        return regions_infs_list
            
    def test_get_contact_mean(self, position):
        return np.zeros(self.dim*2)

    def test_init_value_function(self,phi_inf_list):
        """Initialization of the approximate value function with the amount of regions.
        """
        self.region_amount = len(phi_inf_list)
        self.phi_inf_list = phi_inf_list
        self.state_value_func.init_region_inf(self.region_amount, self.demo_act_dict, phi_inf_list)

    def test_learn_initial_policy(self):
        """Learn the policy with the experience tuple.
        """
        states,actions,rewards,next_states,dones = self.memory.batch_sample()
        loss = self.state_value_func.q_learn(states,actions,rewards,next_states,dones)
        return loss
    
    def get_region_q_value(self):
        return self.state_value_func.get_region_q()
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
