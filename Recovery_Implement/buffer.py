import region

class Exp_Buffer:
"""Buffer initialization

Parameters
----------
memory : (list)
	Restores the experience namedtuples.
experience : (namedtuple)
	Restores the experience with name(label).

Return
------
"""
    def __init__(self,batch_size):
        self.memory = []
        self.batch_size = batch_size

        self.experience = namedtuple("Experience", \
                                 field_name = ["state", "contact", "action", "reward", "next_state", "next_contact", "done"])
        self.cluster = Region_Cluster()
        self.region_psi_result_set = set()
    def add(self, state, contact, action, reward, next_state, next_contact, done):
        """Add a new experience to memory."""
        e = self.experience(state, contact, action, reward, next_state, next_contact, done)
        self.memory.append(e)

"""Buffer initialization

Parameters
----------
memory : (list)
	Restores the experience namedtuples.
experience : (namedtuple)
	Restores the experience with name(label).

Return
------
Tuple: (states, contacts, actions, rewards, next_states,next_contacts, dones)
Each of the element is tensor with shape (batch_size,1)
"""
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
		# np.vstack: reshape list to ndarray, column shape (n,1).
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        contacts = torch.from_numpy(np.vstack([e.contact for e in experiences if e is not None])).long().to(device)

        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)\
		action_list = []
		for e in experiences if e is not None:
			action = int(list(e.action.keys())[0])
			action_list.append(action)
		actions = torch.from_numpy(np.vstack(action_list))

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_contacts = torch.from_numpy(np.vstack([e.next_contact for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, contacts, actions, rewards, next_states,next_contacts, dones)
# return a list of namedtuple for experience.
    def get_experience_list(self):
        experience_list = self.memory
        return experience_list


"""Clustering a bunch of points into regions.

Parameters
----------
action_list : (list)
	a list of demonstration action

Return
------
region_psi_resultset: (set)
    a set of regions, the result of clusterring and merging.
    (psi_set, psi_set, ...)
    psi_set = set((s,z,a,r,s',z'),...)
"""

    def cluster_region(self, action_dict):
        region_psi_resultset = self.cluster.learn_state_region(self, self.memory, action_dict)
        return region_psi_resultset
