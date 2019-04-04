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
    def __init__(self,batch_size = 4):
        self.memory = []
        self.seperate_memory = []
        self.batch_size = batch_size

        self.seperate_experience = namedtuple("seperate_Experience", \
                                 field_names = ["state", "contact", "action", "reward", "next_state", "next_contact", "done"])

        self.experience = namedtuple("Experience", \
                                 field_names = ["state", "action", "reward", "next_state", "done"])
        self.region_psi_result_set = set()

    def add(self, state,  action, reward, next_state,  done):
        """Add a new experience to memory."""
        e = self.experience(state,  action, reward, next_state, done)
        self.memory.append(e)

    def seperate_add(self, state,contact,  action, reward, next_state,  next_contact, done):
        """Add a new experience to memory."""
        e = self.seperate_experience(state, contact, action, reward, next_state, next_contact, done)
        self.seperate_memory.append(e)

    def batch_sample(self):
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

    def get_seperate_exp_list(self):
        return self.seperate_memory