from collections import namedtuple, deque
import random
import numpy as np
Transition = namedtuple("Transition", ["state", "action", "next_state", "next_action", "reward", "is_terminated", "is_truncated"])


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque([], maxlen)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
