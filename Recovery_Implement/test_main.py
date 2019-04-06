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
import matplotlib.pyplot as plt

def training(agent):
        epoches = 100
        max_t = 1000
        training_loss=[]
        for i_epoch in range(epoches):
                total_loss = 0 
                for t in range(max_t):
                        loss = agent.test_learn_initial_policy()
                        total_loss += loss
                        print('\rEpisode {}\tloss: {:.5f}'.format(i_epoch, loss),end = "")
                training_loss.append(total_loss)
                print('\rEpisode {}\tloss: {:.5f}'.format(i_epoch, loss))
        print('\n',agent.get_funnels_q_value())
        return training_loss

robot = env_robot.Env(dim=2)
agent = Recovery_RL_Agent.Agent(dim=2)

demo_states_list =  [[100,50],[200,200],[300,400],[400,350],[450,450],[500,550]]
robot.demonstration(demo_states_list)
demo_act_dict = agent.demo_record(demo_states_list)

repeat_times = 100

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    episode_record, separate_s_c_record = robot.execute_separate_demo_act(demo_act_dict)
    # episode_record = robot.execute_demo_act(demo_act_dict)
    
    agent.exp_record(episode_record)
    # agent.separate_exp_record(separate_s_c_record)
    # Reset env, back to start point
    robot.test_reset()
# print(agent.get_exp_list())
funnels_infs_list = agent.test_generate_funnels_infs_list()

agent.test_init_value_function(funnels_infs_list,init_Q_value = -1200)
training_loss = training(agent)
agent.test_init_value_function(funnels_infs_list,init_Q_value = (-np.inf))
inf_training_loss = training(agent)


figsize = 15,9
figure, ax = plt.subplots(figsize=figsize)


plt.plot(np.arange(1, len(training_loss)+1), training_loss, color='orangered' )
plt.plot(np.arange(1, len(inf_training_loss)+1), inf_training_loss, color='blue' )

plt.ylabel('training_loss')
plt.xlabel('Episode #')
plt.savefig('training_loss_comparation_1200&inf.png')
plt.show()