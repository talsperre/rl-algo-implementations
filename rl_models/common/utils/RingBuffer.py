import time
import random
import collections
import numpy as np
random.seed(42)
np.random.seed(42)


class RingBuffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = collections.deque(maxlen=max_len)
    
    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled = zip(*[self.buffer[idx] for idx in indices])
        return sampled
