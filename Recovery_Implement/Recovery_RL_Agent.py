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
        self.demo_act_dict = OrderedDict()
        self.num_of_demo_goal = 0
        self.demo_goal = []
        self.q_func = Value_Function(lr=lr, gamma = gamma,init_Q_value = init_Q_value, if_soft_update = if_soft_update,tau = tau)
        self.memory = Exp_Buffer(batch_size= self.batch_size)

        self.regions_dict = OrderedDict()
        # The dimension of the workspace (2 or 3).
        self.dim = dim
#         self.cluster = Region_Cluster()
        self.final_goal_state = 0
        """
        self.act for new skills
        """
        
        self.actions_dict = OrderedDict()
        # the state dim contain the position and contact
        self.state_size = (dim+dim*2)
        self.action_size = 0
        self.seed = random.seed(seed)

        self.funnels_infs_list=[]
        
        self.funnels_amount= 0
        self.separate_s_c_exp_list = []

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
        self.final_goal_state = goal_tuples[-1]

        print("Agent: agent has recorded the demonstration as its original skills: {} \n".format(self.demo_act_dict))
        print("Agent: agent  recorded the final goal state as : {} \n".format(self.final_goal_state))
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
    
    def separate_exp_record(self,episode_list):
        """Record experience tuples of an episode to experience list
        Param
        ----------
        episode_list:(list)
            a list of experience tuple.
        """
        for exp_tuple in episode_list:
            state,contact,  action, reward, next_state,  next_contact, done = exp_tuple
            self.memory.separate_add( state,contact,  action, reward, next_state,  next_contact, done)

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

    def get_separate_exp_list(self):
        """Return the list of all the restored experience tuples
        Return
        ----------
        experiences_list:(list)
            a list of experience tuple.
        """
        return self.memory.get_separate_exp_list()


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


    def test_generate_funnels_infs_list(self, add_stuck_funnels):
        cov = np.eye(self.dim * 3)
        regions = []
        origional_point =  np.zeros(self.dim)
        regions.append(origional_point)
        for each_goal in self.demo_goal:
            regions.append(each_goal)

        self.funnels_amount = len(regions)

        funnels_infs_list=[]
        for region_position, (key,value) in zip(regions,self.demo_act_dict.items()):
            
            if (region_position==np.array(self.final_goal_state)).all():
                break
            alist = []
            act_dict = {}
            act_dict[key] = value
            alist.append(key)
            alist.append(act_dict)

            # Generate the mean covering the position and contact
            contact_mean = self.test_generate_contact_mean(region_position)
            mean = np.append(region_position, contact_mean)

            region_dict = {}
            region_dict[key] = (mean,cov)

            alist.append(region_dict)

            funnels_infs_list.append(alist)

# ----------------------define stuck position -----------------------------------------------
        if add_stuck_funnels:
            stuck_position = np.array((10,10))
# ----------------------generate stuck funnels -----------------------------------------------
            region_position = stuck_position
            alist = []
            act_dict = {}
            act_dict['1'] = [0,20]
            # Append funnel index
            alist.append('3')
            alist.append(act_dict)

            contact_mean = self.test_generate_contact_mean(region_position)
            mean = np.append(stuck_position, contact_mean)
            region_dict = {}
            region_dict['3'] = (mean,cov)
            alist.append(region_dict)
            funnels_infs_list.append(alist)
# # ----------------------generate stuck funnels -----------------------------------------------
            region_position = stuck_position
            alist = []
            act_dict = {}
            act_dict['2'] =[0,30]
            # Append funnel index
            alist.append('4')
            alist.append(act_dict)

            contact_mean = self.test_generate_contact_mean(region_position)
            mean = np.append(stuck_position, contact_mean)
            region_dict = {}
            region_dict['3'] = (mean,cov)
            alist.append(region_dict)
            funnels_infs_list.append(alist)
# # -------------------------------------------------------------------------------------------------

        print("Agent: test funnels infs:{} \n".format(funnels_infs_list))
        return funnels_infs_list
            
    def test_generate_contact_mean(self, position):
        return np.zeros(self.dim*2)

    def test_init_value_function(self,funnels_infs_list):
        """Initialization of the approximate value function with the amount of funnels.
        """
        self.funnels_amount = len(funnels_infs_list)
        self.funnels_infs_list = funnels_infs_list
        # Initialize the q function with th funnels information.
        self.q_func.init_funnels_inf(self.funnels_amount, self.demo_act_dict, funnels_infs_list )

    def test_learn_initial_policy(self, learn_method ='approximation', ordered_sample = False):
        self.ordered_sample = ordered_sample
        """Learn the policy with the original experience tuple.
        """
        if learn_method == 'approximation' :
            states,actions,rewards,next_states,dones = self.memory.batch_sample(learn_method=learn_method, ordered_sample = ordered_sample)
            loss = self.q_func.q_learn(states,actions,rewards,next_states,dones)
            return loss

        elif learn_method == 'Q table'  :
            experience = self.memory.batch_sample(learn_method=learn_method, ordered_sample = ordered_sample)
            self.q_func.q_table_learn(experience)
            return self.q_func.print_Q_table()




    def get_funnels_q_value(self):
        return self.q_func.get_funnels_q()



    def choose_act(self, state, action, recovery_prob = 0):
        """Call the gaussian maximum likelihood estimation method.

        Return
        ------

        """
        pass

