import torch
import numpy as np
from scipy.stats import multivariate_normal
import torch
import torch.nn.functional as F
import torch.optim as optim
LR = 0.01


class Value_Function():
    def __init__(self):
        return
    def init_funnels_inf(self, funnels_amount, act_dict, funnels_infs_list,init_Q_value):

        
        # init q parameters. The input funnels do not contain the beginning one, so add 1 in the amount.
        # self.q = torch.ones(1,funnels_amount) * (-1200)

        self.q = torch.randn(1,funnels_amount)
        self.q.requires_grad = True
        # act inf
        self.demo_act_dict = act_dict
        self.action_amount = len(act_dict)

        self.funnels_infs_list = funnels_infs_list
        self.init_Q_value = init_Q_value


    def qforward(self,states):
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
                for i, funnel_inf_list in enumerate(self.funnels_infs_list):
                    cach = 0 
                    if cmp(a_dict,funnel_inf_list[1]):
                        cach = 0
                    else:
                        #  should be changed to (state, funnel_inf_list[2].values()[0],  funnel_inf_list[2].values()[1])
                        #  The second and third terms are the mean and covariance.
                        phi_state_pro = multivariate_normal.pdf(state, funnel_inf_list[2].values()[0][0], funnel_inf_list[2].values()[0][1])
                        phi_state_pro = torch.from_numpy(np.array(phi_state_pro)).float()
                        cach = self.q[0][i] * phi_state_pro

                    numerator = numerator + cach
    #             compute the Q denominator
                for i, funnel_inf_list in enumerate(self.funnels_infs_list):
                    d_cach = 0
                    if cmp(a_dict,funnel_inf_list[1]):
                        d_cach = 0
                    else:
                        #  should be changed to (state, funnel_inf_list[2].values()[0],  funnel_inf_list[2].values()[1])
                        #  The second and third terms are the mean and covariance.
                        # phi_state_pro = multivariate_normal.pdf(state, funnel_inf_list[2], funnel_inf_list[3])
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

    def get_funnels_q(self):
        return self.q