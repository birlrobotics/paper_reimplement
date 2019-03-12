"""descibision of a function

Parameters
----------
arg1 : ((type))
	descibision
arg2 : (type)
	descibision

Return
------
arg1 : (type)
	descibision
"""
"""
"""
from copy import deepcopy
import gaussian_tool as gt
import buffer
class Agent():

    def __init__(self, state_size, action_size, seed):

        """
        self.demo_act = {a_i: goal_postion}
        """
        self.demo_act_dict = {}
        self.num_of_demo_goal = {}
		self.state_value_func = Value_Function()
        self.exps_list = Exp_Buffer()
        """
        self.act for new skills
        """
        self.actions_dict = {}
        self.state_size = 0
        self.action_size = 0
        self.seed = random.seed(seed)


        """
        self.region_dict:(dictionary)
        {region_index: phi_set},
        region_index = i
        phi_set = set((s,z,a,r,next_s,next_z),...)
        """
		self.regions_infs_list=[]
        self.region_dict={}
        self.num_of_region = 0

    """To record a task demonstration skill with several goal states,
        i.e. the goal 3D position of each initial skill.
        Use ROS interface to get the position information of robot.

    Parameters
    ----------
	goal_tuples:(a list of tuple)

    """
    def demo_record(self,goal_tuples):
        for i, goal in enumerate(goal_tuples):
			action_name = str(i)
			self.demo_act_dict[action_index] = goal
		demo_act_dict = self.demo_act_dict
        return demo_act_dict

#   Return the demo action list to robot (then robot execute the demo to get experience)
    def get_demo_act_dict(self):
		demo_act_dict = self.demo_act_dict
        return demo_act_dict

# record an episode experience when robot executing the demonstration
# Arg:
#       episode_list: a list of namedtuple experience
    def exp_record(self,episode_list):
        for exp_tuple in episode_list:
            self.exps_list.add(exp_tuple)

# return a list of namedtuple for experience.
    def get_exp_list(self):
        experiences_list = self.exps_list.get_experience_list()
        return experiences_list


    """Call the cluster method to get regions. And assign the relative action to each region.
		Also determine a region whether it is a goal region.
	regions_infs_list = [phi_1, phi_2, ...]
	phi_i = [phi_set, act_dict]
	phi_set = ((state,contact,action,reward,next_state,next_contact),...)
	act_dict = [a_index: goal_postion]
    """
    def learn_cluster_region(self):
        region_phi_result_set = self.exps_list.cluster_region(self.demo_act_dict)
		for i, phi_set in enumerate(region_phi_result_set):
			phi_list = phi_set
			self.regions_infs_list.append([])
			self.regions_infs_list[i].append(deepcopy(phi_set))
			# phi_list.[0].a is  {a_index:goal_position}
			self.regions_infs_list[i].append(deepcopy(phi_list.[0].action))
			# determine a region if it is goal region, by checking its action whether is empty or not.
			if bool(hi_list.[0].action) == False:
				self.regions_infs_list[i].append(True)
			else:
				self.regions_infs_list[i].append(False)

		self.num_of_region = len(region_phi_result_set)


	#
    # """Indexing the region, and restore it with region states.
	#
	#
    # Parameters
    # ------
    # self.region_dict:(dictionary)
    # {region_index: phi_set},
    # region_index = i
    # phi_set = set((s,z,a,r,next_s,next_z),...)
	#
    # """
    # def region_indexing(self):
    #     for action in self.demo_act
    #         for phi_set in self.region_phi_result_set
    #         #   convert to list for checking action
    #             list_phi_set = list(phi_set)
    #             if list_phi_set.[0].a == action:
    #                 self.region_dict[self.num_of_region] = phi_set
    #                 self.num_of_region  += 1

    """Get the number of regions.

    Return
    ------
    region_number:(int)
    """
    def get_region_number(self):
        region_number = self.num_of_region
        return region_number

    """Call the gaussian maximum likelihood estimation method.
	Put mean and std in each region_infs_list's 3rd and 4th index of place.
    """
    def gaussian_likelihood(self):
        for i, region_infs_list in enumerate(self.regions_infs_list):
            mean, std = gt.gaussian_estimation(region_infs_list[0])
			self.regions_infs_list[i].append(mean)
			self.regions_infs_list[i].append(std)


    """Construct region directed graph.
    """
    def region_directed_graph(self):
        TODO
	def init_value_function(self):
		self.state_value_func.init_region_number(self.num_of_region, self.demo_act_dict)

    def learn_initial_policy(self):
		# get a tuple of ndarrays： （ndarray, ndarray, ...)
		experiences_batch = self.exps_list.sample()
		states, contacts, actions, rewards, next_states, next_contacts, dones = experiences
		# Get max predicted Q values (for next states) from target model
		# max(1)[0] take the maximum value along a given axis=1 (row).
		# Because it do not need to compute the gradient of target, use detach.
        Q_targets_next = self.state_value_func_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model, along row to get the index with actions
        Q_expected = self.state_value_func_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

 		#  not done
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    """Call the gaussian maximum likelihood estimation method.

    Return
    ------
    self.region_prob_list:(list)
    [region_prob_tuple], region_prob_tuple = (region_index,mean,std,is_goal)
    """

    def choose_act(self, state, action, recovery_prob = 0):
        TODO
    def learn_Q_value(self, ):
