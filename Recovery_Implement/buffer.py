import region
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Exp_Buffer():
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
    def __init__(self,batch_size = 2):
        self.memory = []
        self.separate_memory = []
        self.batch_size = batch_size
        self.order = 0
        self.separate_experience = namedtuple("separate_Experience", \
                                 field_names = ["state", "contact", "action", "reward", "next_state", "next_contact", "done"])

        self.experience = namedtuple("Experience", \
                                 field_names = ["state", "action", "reward", "next_state", "done"])
        self.region_psi_result_set = set()
        self.memory_is_reverse = False
    def add(self, state,  action, reward, next_state,  done):
        """Add a new experience to memory."""
        e = self.experience(state,  action, reward, next_state, done)
        self.memory.append(e)

    def separate_add(self, state,contact,  action, reward, next_state,  next_contact, done):
        """Add a new experience to memory."""
        e = self.separate_experience(state, contact, action, reward, next_state, next_contact, done)
        self.separate_memory.append(e)

    def reverse_memory_list(self):
        if self.memory_is_reverse == False:
            self.memory.reverse()
            self.memory_is_reverse = True
        
    def batch_sample(self):



# -------------------------------------------new_q_learn------------------------------------------ --------------------
        experiences = random.sample(self.memory, k=self.batch_size)

        # np.vstack: reshape list to ndarray, column shape (n,1).
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        # contacts = torch.from_numpy(np.vstack([e.contact for e in experiences if e is not None])).long()

        action_list = []
        for e in experiences:
            if e is not None:
                action = int(list(e.action.keys())[0])
                action_list.append(action)
        actions = torch.from_numpy(np.vstack(action_list))

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        # next_contacts = torch.from_numpy(np.vstack([e.next_contact for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        

        return (states, actions, rewards, next_states, dones)

        
# return a list of namedtuple for experience.
    def get_experience_list(self):
        experience_list = self.memory
        return experience_list

    def get_separate_exp_list(self):
        return self.separate_memory