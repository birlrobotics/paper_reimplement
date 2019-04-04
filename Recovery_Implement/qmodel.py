import torch
import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.optim as optim
LR = 0.1


class Value_Function():
    def __init__(self):
        return
    def init_region_inf(self, region_amount, act_dict, phi_inf_list):

        
        # init q parameters. The input regions do not contain the original one, so add 1 in the amount.
        # self.q = torch.ones(1,region_amount) * (-1200)
        self.q = torch.randn(1,region_amount)
        self.q.requires_grad = True
        # act inf
        self.demo_act_dict = act_dict
        self.action_amount = len(act_dict)

        self.regions_infs_list = phi_inf_list



    def qforward(self,states):
        """Q approximation function

        Parameters
        ----------
        state : (ndarray)
        The position of agent. (2D or 3D)
        contact : (ndarray)
        The contact mode of agent. (2D or 3D)
        regions_infs_list: (list)
        A list of regions informations: [phi_1,phi_2,...],
        phi_i = [phi_set, region_action, is_goal, mean, std, father, son]
        region_action = {a_index: goal, ...}
        Return
        ---------
        s_a_q_arrary:(np.ndarray)
        1 dimension array recording each action value of a specific array.
        """
        # init a batch, size: 1 * batch_size * (action+1) dim
        # the 2nd size is not the state dim, because we input a batch of states.
        # We +1 because we assume that in the final state there still an action
        Q_batch_tensor = torch.ones((1,len(states),(self.action_amount+1)))
        for b_number, state in enumerate(states):
            Q_tensor = torch.ones((1,(self.action_amount+1))) * (-1200)
            for demo_a, skill_goal in self.demo_act_dict.items():
                a_dict = {}
                a_dict[demo_a] =skill_goal
                numerator = 0
                denominator = 0
    #             compute the Q numerator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    cach = 0 
                    if cmp(a_dict,region_infs_list[1]):
                        cach = 0
                    else:
                        phi_state_pro = multivariate_normal.pdf(state, region_infs_list[2], region_infs_list[3])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        cach = self.q[0][i] * phi_state_pro

                    numerator = numerator + cach
    #             compute the Q denominator
                for i, region_infs_list in enumerate(self.regions_infs_list):
                    d_cach = 0
                    if cmp(a_dict,region_infs_list[1]):
                        d_cach = 0
                    else:
                        phi_state_pro = multivariate_normal.pdf(state, region_infs_list[2], region_infs_list[3])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        d_cach = phi_state_pro

                    denominator = denominator + d_cach
        #         This "if" is hyperparameter, if it is too small, then the tensor will turn into Nan, which cannot be 
        #         compute the gradient. But if it is too big, then we cannot find a good value between the region 
        #         means.
                if denominator >= 1.0e-20:
                    Q_tensor[0][int(demo_a)] = numerator/denominator
                else:
                    Q_tensor[0][int(demo_a)] = -1200
            Q_batch_tensor[0][b_number] = Q_tensor
        Q_batch_tensor.requires_grad
        return Q_batch_tensor


    def q_learn(self,states,actions,rewards,next_states,dones):
        lr = LR
        # abandon_last_q = self.qforward(next_states).squeeze()[:,0:-1]
        # Q_target_next = abandon_last_q.max(1)[0]

        Q_target_next = self.qforward(next_states).squeeze().max(1)[0]
        Q_target = rewards + Q_target_next.reshape(-1,1)*(1 - dones)
        Q_expect =  self.qforward(states).squeeze().gather(1,actions.long())
        Q_expect.requires_grad
        loss = F.mse_loss(Q_expect,Q_target)
        loss.backward()

        self.q.data = self.q.data - lr * self.q.grad.data
        # print('\rEpisode {}\tloss: {:.5f}'.format(e, loss),end = "")
        self.q.grad.data.zero_()
        return loss.data

    def get_region_q(self):
        return self.q