from __future__ import print_function
import torch
import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

class Value_Function():
    def __init__(self,lr,gamma,init_Q_value, if_soft_update, tau = 1e-2):
        self.lr =lr
        self.gamma = gamma
        self.tau = tau
        self.init_Q_value = init_Q_value
        self.if_soft_update = if_soft_update
        return
    def init_funnels_inf(self, funnels_amount, act_dict, funnels_infs_list):

        
        # init q parameters. The input funnels do not contain the beginning one, so add 1 in the amount.
        # self.q = torch.ones(1,funnels_amount) * (-1200)

        self.q_local = torch.ones(1,funnels_amount) * (self.init_Q_value)
        self.q_local.requires_grad = True

        self.q_target = torch.ones(1,funnels_amount) * (self.init_Q_value)
        # self.q_target.requires_grad = True

        # act inf
        self.demo_act_dict = act_dict
        self.action_amount = len(act_dict)

        self.funnels_infs_list = funnels_infs_list
        self.funnels_amount = funnels_amount

# -----------------------------table learning-------------------
        self.Q = defaultdict( lambda : np.ones(self.action_amount)* self.init_Q_value)
# -----------------------------table learning-------------------

    def qforward(self,q_params,states):
        """Q approximation function

        Parameters
        ----------
        state (ndarray): 
            the conbination of position and contact.The position of agent (2D or 3D), and the contact mode of agent (2D or 3D).

        funnels_infs_list (list):
            A list of funnels informations (list): [f_1,f_2,...], and
            f_i = [f_index, f_action, mean, std, is_goal, father, son]
            f_action = {a_index: goal}


        Return
        ---------
        Q_batch_tensor(torch.tenser):
        1 dimension array recording each action value of a specific array.
        """
        # init a batch, size: 1 * batch_size * (action_amount) dim
        # the 2nd size is not the state dim, because we input a batch of states.
        Q_batch_tensor = torch.ones((1,len(states),(self.action_amount)))
        for b_number, state in enumerate(states):
            Q_tensor = torch.ones((1,(self.action_amount))) * (self.init_Q_value)
            for demo_a, skill_goal in self.demo_act_dict.items():
                a_dict = {}
                a_dict[demo_a] =skill_goal
                numerator = 0
                denominator = 0
    #             compute the Q numerator
                for i, funnel_inf_list in enumerate(self.funnels_infs_list):
                    cach = 0 
                    if cmp(a_dict,funnel_inf_list[1]):
                        cach = 0
                    else:
                        #  The second and third terms are the mean and covariance.
                        phi_state_pro = multivariate_normal.pdf(state, funnel_inf_list[2].values()[0][0], funnel_inf_list[2].values()[0][1])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        cach = q_params[0][i] * phi_state_pro

                    numerator = numerator + cach
    #             compute the Q denominator
                for i, funnel_inf_list in enumerate(self.funnels_infs_list):
                    d_cach = 0
                    if cmp(a_dict,funnel_inf_list[1]):
                        d_cach = 0
                    else:
                        #  The second and third terms are the mean and covariance.
                        phi_state_pro = multivariate_normal.pdf(state, funnel_inf_list[2].values()[0][0], funnel_inf_list[2].values()[0][1])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        d_cach = phi_state_pro

                    denominator = denominator + d_cach
        #         This "if" is hyperparameter, if it is too small, then the tensor will turn into Nan, which cannot be 
        #         compute the gradient. But if it is too big, then we cannot find a good value between the region 
        #         means.
                if denominator >= 1.0e-20:
                    Q_tensor[0][int(demo_a)] = numerator/denominator
                else:
                    Q_tensor[0][int(demo_a)] = self.init_Q_value
            Q_batch_tensor[0][b_number] = Q_tensor
        Q_batch_tensor.requires_grad
        return Q_batch_tensor


    def q_learn(self, states,actions,rewards,next_states,dones):
        if self.if_soft_update:
            return self.q_learn_softupdate(states,actions,rewards,next_states,dones)
        else:
            return self.q_learn_without_softupdate(states,actions,rewards,next_states,dones)

    def q_learn_without_softupdate(self,states,actions,rewards,next_states,dones):
        Q_target_next = self.qforward(self.q_local,next_states).squeeze().max(1)[0]
        Q_target = rewards + self.gamma * Q_target_next.reshape(-1,1)*(1 - dones)
        Q_expect =  self.qforward(self.q_local,states).squeeze().gather(1,actions.long())
        Q_expect.requires_grad
        loss = F.mse_loss(Q_expect,Q_target)
        loss.backward()

        self.q_local.data = self.q_local.data - self.lr * self.q_local.grad.data
        self.q_local.grad.data.zero_()
        return loss.data

    def q_learn_softupdate(self,states,actions,rewards,next_states,dones):
        Q_target_next = self.qforward(self.q_target,next_states).squeeze().max(1)[0]
        Q_target = rewards + self.gamma * Q_target_next.reshape(-1,1)*(1 - dones)
        Q_expect =  self.qforward(self.q_local,states).squeeze().gather(1,actions.long())
        Q_expect.requires_grad
        loss = F.mse_loss(Q_expect,Q_target)
        loss.backward()

        self.q_local.data = self.q_local.data - self.lr * self.q_local.grad.data
        self.q_local.grad.data.zero_()
        self.soft_update()
        return loss.data

    def get_funnels_q(self):
        return self.q_local

    def soft_update(self):
        self.q_target.data.copy_((1.0-self.tau) * self.q_target.data + self.tau * self.q_local.data)


# # -------------------------Q table----------------------------------------------------------------------------
    def q_table_learn(self,experience):
        state,action,reward,next_state,done = experience
        # state = state.numpy()
        state = self.totuple(state)
        # next_state = next_state.numpy()
        next_state= self.totuple(next_state)
        # Convert the act dict to relative index
        action = int(action.keys()[0])
        if done:
            self.Q[state][action] = self.update_Q_table(self.Q[state][action], 0,reward)
        else:
            self.Q[state][action] = self.update_Q_table(self.Q[state][action], np.max(self.Q[next_state]),reward)


    def update_Q_table(self,Qsa, Qsa_next, reward):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.lr * (reward + (self.gamma * Qsa_next) - Qsa))


    def print_Q_table(self):
       return self.Q


# Convert the tenosr to tuple for the key of dict, otherwise something wired will happen.
    def totuple(self,a):
        try:
            return tuple(self.totuple(i) for i in a)
        except TypeError:
            return a
# # -------------------------Q table----------------------------------------------------------------------------
