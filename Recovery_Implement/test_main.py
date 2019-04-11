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

LR = 1e-2              # learning rate 
BATCH_SIZE = 2          # Only for approximation, the Q table cannot batch update.
GAMMA = 1
HAVE_STUCK = True              # if True, it will include the stuck situation
EPOCHES = 100
MAX_T = 100    
ADD_NOISE = True
ADD_PERT = True                          # If false, all the contact mode is zero.
SOFT_UPDATE = True
INIT_Q_VALUE = (-10000)
TAU = 1e-3


def training(agent,epoches,max_t):
        training_loss=[]
        f = open('q_parameters.txt', 'wb') # python3, write bytes not unicode
# -------------------------Approximation with Stochasitc Gradient--------------------------------------
        for i_epoch in range(epoches):
                total_loss = 0
                for t in range(max_t):
                        loss = agent.learn_initial_policy()
                        total_loss += loss
                        mean_loss = total_loss/(t+1)
                        print('\rEpisode {}\tloss: {:.5f}'.format(i_epoch, loss),end = "")
                        
                training_loss.append(mean_loss)
                # Save q_value
                q_parameters = agent.get_funnels_q_value()
                np.savetxt(f,np.ones(1)* i_epoch,header='epoch',fmt='%d', comments= '')
                np.savetxt(f,q_parameters.detach(),newline='\n',fmt='%1.5f')
                np.savetxt(f,np.ones(1)*mean_loss.detach().numpy(),newline='\n',fmt='%1.5f',header='mean_loss')
                print('\rEpisode {}\tloss: {:.5f}'.format(i_epoch, mean_loss))

        minimum_epoch = training_loss.index(min(training_loss))
        np.savetxt(f,np.ones(1)*minimum_epoch,newline='\n',fmt='%d',comments= '',header='=======minimum loss episode=====')
        print('\rMinimum loss is in epoch {}'.format(minimum_epoch))
        plot_loss(training_loss)
        print('The learned q parameters are :{} \n'.format(q_parameters))

        


def plot_loss(training_loss):
        figsize = 15,9
        figure, ax = plt.subplots(figsize=figsize)
        plt.plot(np.arange(1, len(training_loss)+1), training_loss, color='blue' )
        plt.ylabel('training_loss')
        plt.xlabel('Episode #')
        plt.savefig('softupdate_3*100_randomsamples.png')
        plt.show()


# ------------------------------------------Main{}------------------------------------------

robot = env_robot.Env(dim=2,have_stuck_funnels= HAVE_STUCK, add_noise=ADD_NOISE,add_pert= ADD_PERT,)
agent = Recovery_RL_Agent.Agent(init_Q_value = INIT_Q_VALUE,dim=2,batch_size= BATCH_SIZE, lr=LR ,gamma=GAMMA,if_soft_update = SOFT_UPDATE, tau = TAU)
demo_position_list =  [[0,10],[0,20],[0,30]]
robot.demonstration(demo_position_list)
demo_act_dict = agent.demo_record(demo_position_list)

# ------------------------------------------Generate exp tuples------------------------------------------
repeat_times = 100

for i in range(0,repeat_times):
    # executing the demo action and restore experience tuples in agent
    episode_record = robot.execute_demo_act(demo_act_dict)
    
    agent.exp_record(episode_record)
    # Reset env, back to start point
    robot.test_reset()
    
# for e in agent.get_exp_list():
#         print(e.state,e.action, e.reward, e.next_state, e.done)
# ----------------------------------------------------------------------------------------------------------


# ------------------------------------------Get Funnels information------------------------------------------
agent.learn_funnels_infs()

agent.init_value_function()
# ----------------------------------------------------------------------------------------------------------




training(agent, epoches = EPOCHES, max_t = MAX_T)
