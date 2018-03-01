import random
import numpy as np
from collections import namedtuple, defaultdict

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def state_action_counts(self):
        freqs = defaultdict(lambda: defaultdict(int))
        for transition in self.memory:
            state = transition.state[0].numpy()
            state = np.argmax(state)
            action = transition.action[0,0]
            freqs[state][action] += 1
        return freqs

    def __len__(self):
        return len(self.memory)