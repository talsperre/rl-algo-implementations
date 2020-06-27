import time
import random
import collections
import numpy as np

from tqdm import tqdm
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

r_buffer = RingBuffer(200000)
for i in tqdm(range(10000)):
    states = np.random.randn(4, 84, 84)
    next_states = np.random.randn(4, 84, 84)
    rewards = np.random.randn()
    dones = np.random.randn()
    actions = np.random.randn()
    r_buffer.append((states, actions, rewards, dones, next_states))

for i in range(100):
    st_time = time.time()
    sampled = r_buffer.sample(32)
    print(time.time() - st_time)