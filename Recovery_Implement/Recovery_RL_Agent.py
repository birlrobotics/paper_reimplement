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
from region import Region_Cluster
from scipy.special import softmax

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class Agent():

    def __init__(self, init_Q_value=0 ,seed = 0, dim =2 , batch_size = 2, lr = 0.1 , gamma=1,tau = 1e-2, if_soft_update =True  ):

        """
        param
        -----
        seed(int): random seed
        dim(int): the dimension of position space (not include the contact dim.)
        batch_size(int):
        lr(float):learning rate
        gamme(float):the discount factor

        """
        # ordered_sample to determine if sequentially draw the sample to update or not
        self.batch_size = batch_size
        self.ordered_sample = False

        self.lr = lr
        self.demo_acts_dict = OrderedDict()
        self.num_of_demo_goal = 0
        self.demo_goal = []
        self.gamma = gamma
        self.tau = tau
        self.if_soft_update = if_soft_update
        self.init_Q_value = init_Q_value
        self.memory = Exp_Buffer(batch_size= self.batch_size)
        self.regions_dict = OrderedDict()
        # The dimension of the workspace (2 or 3).
        self.dim = dim
        self.final_goal_state = 0
        """
        self.act for new skills
        """
        self.actions_dict = OrderedDict()
        # the state dim contain the position and contact
        self.state_size = (dim+dim*2)
        self.action_size = 0
        self.seed = random.seed(seed)

        # Funnel operation variables
        self.funnels = Region_Cluster(dim=self.dim)
        self.funnels_inf_list=[]
        self.funnels_amount= 0

    def demo_record(self,goal_tuples):
        """To record a task demonstration skill with several goal states,
        i.e. the goal 3D position of each initial skill.
        Use ROS interface to get the position information of robot.

        Parameters:
            goal_tuples(list):a list of 2 or 3 dim array.
        """
        self.demo_goal = goal_tuples
        self.demo_amount = len(goal_tuples)
        for i, goal in enumerate(goal_tuples):
            action_index = str(i)
            self.demo_acts_dict[action_index] = goal
        self.final_goal_state = goal_tuples[-1]

        print("Agent: agent has recorded the demonstration as its original skills: {} \n".format(self.demo_acts_dict))
        print("Agent: agent  recorded the final goal state as : {} \n".format(self.final_goal_state))
        return self.demo_acts_dict

    def get_demo_acts_dict(self):
        """Return the action dictionary of demonstration
        Return
        ----------
        demo_acts_dict:(dict)
            keys(string): index 
            values(tuple): goal position
        """
        self.demo_acts_dict
        return self.demo_acts_dict

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

    def learn_funnels_infs(self):
        experiences_list = self.memory.get_experience_list()
        self.funnels_inf_list = self.funnels.learn_funnels(experiences_list, self.demo_acts_dict)

    def get_funnels_amount(self):
        """Get the amount of funnels.
        Return
        ------
        self.funnels_amount(int):
        """
        pass
        return self.funnels_amount



    def region_directed_graph(self):
        """Construct region directed graph used for algorithm 2.
        """
        pass

    def init_value_function(self):
        """Initialization of the approximate value function with the amount of funnels.
        """
        self.funnels_amount = len(self.funnels_inf_list)

        self.qlearning_method = Value_Function(lr=self.lr, demo_acts_dict = self.demo_acts_dict, \
                                        funnels_inf_list = self.funnels_inf_list,gamma = self.gamma,\
                                        init_Q_value = self.init_Q_value, if_soft_update = self.if_soft_update,tau = self.tau)



    def learn_initial_policy(self):
        """Learn the policy with the original experience tuple.
        """
        experiences = self.memory.batch_sample()
        # step
        loss = self.qlearning_method.q_learn(experiences)
        self.qlearning_method.soft_update()
        return loss

    def choose_act(self,state):
        Q_s_tensor = self.qlearning_method.forward(state)
        action_prob_vector = softmax(Q_s_tensor)
        np.random.choice(demo_acts_dict,p = action_prob_vector)


        Qmax, argmax_a_index = Q_s_tensor.max(0)

        return_a_dict = {}
        return_a_dict[argmax_a_index] = self.demo_acts_dict[argmax_a_index]

        return return_a_dict

    def get_funnels_q_value(self):
        return self.qlearning_method.get_param()



