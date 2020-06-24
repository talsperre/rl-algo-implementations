import random
random.seed(42)


class RingBuffer(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = []
        self.head = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, elem):
        if len(self.buffer) < self.max_len:
            self.buffer.append(elem)
        else:
            self.buffer[self.head] = elem
        self.head = (self.head + 1) % self.max_len
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
