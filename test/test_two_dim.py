from __future__ import print_function
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import norm
from scipy.stats import multivariate_normal
from collections import namedtuple, OrderedDict
from math import sqrt
import random
# learning rate

LR = 0.1
class two_dim_test():
    def __init__(self):
        self.current_pos = [0,0]
        self.final_goal_state = [500,550]
        self.q = torch.tensor((-1100.,-1100.,-1100.,-1100.,-1100.,-1100.,-1100.)).view(1,7)
        self.q.requires_grad = True
        self.exp = namedtuple("e",field_names = ["state", "action", "reward", "next_state", "done"])

    def state_distance(self,p1,p2):
        distance = 0
        for i,j in zip(p1,p2):
            distance += (i - j)**2
        distance = sqrt(distance)
        return distance

    def move_to(self,goal):
        # move the arm to goal position
        # Noise: mean =0 , std = 1, dim = 2
        self.current_pos = goal + np.random.normal(0,1,2)
        # self.current_pos = goal
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

    def two_dim_sample_gen(self):
        # Execute the demonstration to get exp tuple.
        
        episode_record= []
        state = self.current_pos
        for a_i, a_goal in self.act_dict.items():
            action = a_i
            next_state = self.move_to(a_goal)
            reward, done = self.reward_done(state,next_state)
            
            exp_tuple = self.exp(state,action, reward, next_state,done)
            episode_record.append(exp_tuple)
            state = next_state
            
        return episode_record

    def get_batch_sample(self,exp_list,batch_num):
        # sample a batch of exp tuples and seperate them into column vector.
        batch_list = random.sample(exp_list,batch_num)
        # Convert the type into torch tensor type for calculation.
        states = torch.from_numpy(np.vstack([e.state for e in batch_list if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in batch_list if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch_list if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch_list if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in batch_list if e is not None]).astype(np.uint8)).float()
        return states,actions,rewards,next_states,dones

    def reset_pos(self):
        # Call it after each episode execution.
        self.current_pos = [0,0]



    def qforward(self,states):

        Q_batch_tensor = torch.ones((1,len(states),7))
        for b_number, state in enumerate(states):
            Q_tensor = torch.ones((1,7)) * (-1100)
            for demo_a in self.act_dict.keys():
                numerator = 0
                denominator = 0
    #             compute the Q numerator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    cach = 0 
                    if demo_a != region_infs_list[2]:
                        cach = 0
                    else:
                        phi_state_pro = multivariate_normal.pdf(state, region_infs_list[0], region_infs_list[1])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        cach = self.q[0][i] * phi_state_pro
                    numerator = numerator + cach
    #             compute the Q denominator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    d_cach = 0
                    if demo_a != region_infs_list[2]:
                        d_cach = 0
                    else:
                        phi_state_pro = multivariate_normal.pdf(state, region_infs_list[0], region_infs_list[1])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        d_cach = phi_state_pro

                    denominator = denominator + d_cach
        #         This "if" is hyperparameter, if it is too small, then the tensor will turn into Nan, which cannot be 
        #         compute the gradient. But if it is too big, then we cannot find a good value between the region 
        #         means.
                if denominator >= 1.0e-20:
                    Q_tensor[0][demo_a] = numerator/denominator
                else:
                    Q_tensor[0][demo_a] = -1100
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

    def init_region_infs(self):
        self.demo_goal = [[0,0],[100,50],[200,200],[300,400],[400,350],[450,450],[500,550]]
        means = self.demo_goal
        cov = [[5,0],[0,5]]
        self.regions_infs_list = []
        for i, mean in enumerate(means):
            alist = []
            alist.append(mean)
            alist.append(cov)
            alist.append(i)
            self.regions_infs_list.append(alist)
            
        self.act_dict = OrderedDict()

        for i,region in enumerate(self.regions_infs_list):
        #     The final region doesn't have any action.
            if i ==6:
                break
            self.act_dict[self.regions_infs_list[i][2]] = self.regions_infs_list[i+1][0]
            
        print(self.act_dict.items())
        # print(self.regions_infs_list)