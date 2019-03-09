class Agent():

    def __init__(self, state_size, action_size, seed):

        self.demo_act = {}
        self.num_of_demo_goal = {}

        self.experience = Exp_Buffer()

        self.act = {}
        self.state_size = 0
        self.action_size = 0
        self.seed = random.seed(seed)

#   To record a task demonstration skill with several goal states,
#   i.e. the goal 3D position of each initial skill.
    def demo_record(self):
        self.num_of_demo_goal =
        self.demo_act =

        return

#   Return the demo action list to robot (then robot execute the demo to get experience)
    def get_demo_act_list(self):
        demo_act_list = self.demo_act
        return demo_act_list

# record an episode experience when robot executing the demonstration
# Arg:
#       episode_list: a list of namedtuple experience
    def exp_record(self,episode_list):
        for exp_tuple in episode_list:
            self.experience.append(exp_tuple)

# return a list of namedtuple for experience.
    def get_exp_list(self):
        exp_list = self.experience.get_experience_tuple()
        return exp_list

# ================================================================================================
class Exp_Buffer():

    def __init__(self):
        self.memory = []
        self.experience = namedtuple("Experience", \
                                 field_name = ["s", "z", "a", "r", "next_s", "next_z", "done"])


    def add(self, state, contact, action, reward, next_state, next_contact, done):
        """Add a new experience to memory."""
        e = self.experience(state, contact, action, reward, next_state, next_contact, done)
        self.memory.append(e)

#  Not done yet
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.s for e in experiences if e is not None])).float().to(device)
        contact = torch.from_numpy(np.vstack([e.z for e in experiences if e is not None])).long().to(device)
        action = torch.from_numpy(np.vstack([e.a for e in experiences if e is not None])).float().to(device)
        reward = torch.from_numpy(np.vstack([e.r for e in experiences if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([e.next_s for e in experiences if e is not None])).float().to(device)
        next_contact = torch.from_numpy(np.vstack([e.next_z for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
# return a list of namedtuple for experience.
    def get_experience_list(self):
        experience_list = self.memory
        return experience_list
