from math import sqrt
import numpy as np
class Env():

    def __init__(self,dim):
        self.current_pos = [0]*dim
        self.dim = dim
#   Reset the environment to initial state.

    def reset(self):
        self.current_pos = [0]*self.dim

    def get_reward(self, state, next_state):
        r =0
        print(state,next_state)
        for i,j in zip(state,next_state):
            r += abs(i**2 - j**2)
        return -sqrt(r)

    def test_pertubation(self,  delta = 10):
        per = np.random.normal(0, delta, 2*self.dim)
        return per

    def test_robot_move(self, goal,  mean = 0, std = 12):
        self.current_pos = goal + np.random.normal(mean, std, self.dim)

    def get_pos(self):
        return self.current_pos

    def perturbation(self, delta_d):
        contact_mode = []

        current_s = ROS_current_pos()

        desired_1 = current_s + delta_d
        desired_2 = current_s - delta_d
        desired_list = [desired_1,desired_2]

        for desired_s in desired_list:

            ROS_move_to(desired_s)
            result_s = ROS_current_pos() - current_s
            contact_s = result_s/delta_d
            contact_mode.append(contact_s)
            ROS_move_to(current_s)

        return contact_mode

# robot executes the demo action
# arg:
#       execute_demo_act_list: input a list of action for execution.
# return:
#       episode_record: An episode experience. Restored as a list of
#       namedtuple [(s,z,a,r,s',z',),(s,z,a,r,s',z',),(s,z,a,r,s',z',),...]

    def execute_demo_act(self,execute_demo_act_dict):

        episode_record= []
        cache_exp_tuple = ()

        state = self.current_pos
        contact = self.test_pertubation()

        for  act_index, act_goal in execute_demo_act_dict.items():

#           Noise move means adding the Gaussian noise to the goal position of an action,
#           to model the mechanical or control error.
            action={}
            action[act_index] = act_goal
            self.test_robot_move(act_goal)

            next_state = self.current_pos
            next_contact = self.test_pertubation()
            reward = self.get_reward(state, next_state)

            exp_tuple = cache_exp_tuple = (state, contact, action, reward , next_state, next_contact)
            episode_record.append(exp_tuple)

            state = next_state
            contact = next_contact

        return episode_record
