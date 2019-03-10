class Env():

    def __init__():

        self.m

#   Reset the environment to initial state.

    def reset():

    def get_reward():

        return r

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

    def execute_demo_act(self,execute_demo_act_list):

        episode_record= []
        cache_exp_tuple = ()

        state = ROS_current_pos
        contact = perturbation()

        for i, act in enumerate(execute_demo_act_list):

#           Noise move means adding the Gaussian noise to the goal position of an action,
#           to model the mechanical or control error.

            ROS_noise_move_to(act)

            next_state = ROS_current_pos
            next_contact = perturbation()
            reward = get_reward()

            exp_tuple = cache_exp_tuple = (state, contact, reward, act, next_state, next_contact)
            episode_record.append(exp_tuple)

            state = next_state
            contact = next_contact

        return episode_record
