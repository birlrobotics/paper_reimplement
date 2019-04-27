from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm
from scipy.stats import multivariate_normal
from collections import namedtuple
from math import sqrt
import random

class one_dim_test():
    def __init__(self):
        self.current_pos = 0
        self.final_goal_state = 300
        self.q = torch.tensor((-800.,-800.,-800.,-800.)).view(1,4)
        self.q.requires_grad = True
        self.demo_act = [0,1,2]
        self.regions_infs_list = [[0,5,0],[100,5,1],[200,5,2],[300,5,3]]
        self.demo_goal = [100,200,300]

    def state_distance(self,p1,p2):
        distance = 0
    #     for i,j in zip(p1,p2):
        distance += (p1 - p2)**2
        distance = sqrt(distance)
        return distance

    def move_to(self,goal):
        # move the arm to goal position
        # self.current_pos = goal + np.random.normal(0, 10)
        self.current_pos = goal
        return self.current_pos

    def reward_done(self,state,next_state):
        #  get the reward, and determine whether it is get to goal state.
        distance = self.state_distance(state,next_state)
        r= distance
        if self.state_distance(self.final_goal_state, next_state) > 20:
            r += (.2 * r)
        else:
            return -r, True
        return -r, False

    def one_dim_sample_gen(self):
        # Execute the demonstration to get exp tuple.
        exp = namedtuple("e",field_names = ["state", "action", "reward", "next_state", "done"])
        episode_record= []
        state = self.current_pos
        for goal in self.demo_goal:
            if goal == 100:
                action = 0
            elif goal == 200:
                action = 1
            else:
                action = 2
            next_state = self.move_to(goal)
            reward, done = self.reward_done(state,next_state)
            
            exp_tuple = exp(state,action, reward, next_state,done)
            episode_record.append(exp_tuple)
            state = next_state
            
        return episode_record

    def get_batch_sample(self,exp_list,batch_num):
        batch_list = random.sample(exp_list,batch_num)
        # Convert the type into torch tensor type for calculation.
        states = torch.from_numpy(np.vstack([e.state for e in batch_list if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in batch_list if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch_list if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch_list if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in batch_list if e is not None]).astype(np.uint8)).float()
        return states,actions,rewards,next_states,dones

    def reset_pos(self):
        self.current_pos = 0



    def qforward(self, states):
        
        Q_batch_tensor = torch.ones((1,len(states),4))
        for b_number, state in enumerate(states):
            Q_tensor = torch.ones((1,4)) * (-1000)
            for demo_a in self.demo_act:
                numerator = 0
                denominator = 0
    #             compute the Q numerator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    cach = 0 
                    if demo_a != region_infs_list[2]:
                        cach = 0
                    else:
                        phi_state_pro = norm.pdf(state, region_infs_list[0], region_infs_list[1])
                        phi_state_pro = torch.from_numpy(phi_state_pro).float()
                        cach = self.q[0][i] * phi_state_pro
                    numerator = numerator + cach
    #             compute the Q denominator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    d_cach = 0
                    if demo_a != region_infs_list[2]:
                        d_cach = 0
                    else:
                        phi_state_pro = norm.pdf(state, region_infs_list[0], region_infs_list[1])
                        phi_state_pro = torch.from_numpy(phi_state_pro).float()
                        d_cach = phi_state_pro

                    denominator = denominator + d_cach
        #         This "if" is hyperparameter, if it is too small, then the tensor will turn into Nan, which cannot be 
        #         compute the gradient. But if it is too big, then we cannot find a good value between the region 
        #         means.
                if denominator >= 1.0e-20:
                    Q_tensor[0][demo_a] = numerator/denominator
                else:
                    Q_tensor[0][demo_a] = -1000
            Q_batch_tensor[0][b_number] = Q_tensor
        return Q_batch_tensor

    def q_learn(self,states,actions,rewards,next_states,dones, e):
        lr = 0.1
        Q_target_next = self.qforward(next_states).squeeze().max(1)[0]
        Q_target = rewards + Q_target_next.reshape(-1,1)*(1 - dones)
        Q_expect =  self.qforward(states).squeeze().gather(1,actions.long())
        
        loss = F.mse_loss(Q_expect,Q_target)
        loss.backward()
        
        self.q.data = self.q.data - lr * self.q.grad.data
        print('\rEpisode {}\tloss: {:.5f}'.format(e, loss),end = "")
        self.q.grad.data.zero_()

    def get_parameters(self):
        print(self.q)