from __future__ import print_function
import Recovery_RL_Agent
import env_robot
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm
from scipy.stats import multivariate_normal
from math import sqrt
import random
from math import log


robot = env_robot.Env(dim=2)
agent = Recovery_RL_Agent.Agent()

demo_goal =  [[100,50],[200,200],[300,400],[400,350],[450,450],[500,550],[500,650]]
robot.demonstration(demo_goal)
demo_act_dict = agent.demo_record(demo_goal)

repeat_times = 1

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    # episode_record, seperate_s_c_record = robot.execute_demo_act(demo_act_dict)
    episode_record = robot.execute_demo_act(demo_act_dict)
    
    agent.exp_record(episode_record)
    # agent.seperate_exp_record(seperate_s_c_record)
    # Reset env, back to start point
    robot.test_reset()
print(agent.get_exp_list())
phi_inf_list = agent.test_get_phi_inf_list()

agent.test_init_value_function(phi_inf_list)

# epoches = 10
# max_t = 1000
# for i_epoch in range(epoches):
#     for t in range(max_t):
#         loss = agent.test_learn_initial_policy()
#         print('\rEpisode {}\tloss: {:.5f}'.format(i_epoch, loss),end = "")

# print('\n',agent.get_region_q_value())