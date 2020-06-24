import sys
sys.path.append("../")
import random
import unittest
import rl_models

from rl_models.common.utils.RingBuffer import RingBuffer
random.seed(42)


class TestRingBuffer(unittest.TestCase):
    def test_len(self):
        exp_replay = RingBuffer(3)
        data = [(1, 1), (2, 2)]
        for i in range(len(data)):
            exp_replay.append(data[i])
        self.assertEqual(len(exp_replay), 2)

    def test_append(self):
        exp_replay = RingBuffer(3)
        data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        for i in range(len(data)):
            exp_replay.append(data[i])
        self.assertEqual(exp_replay.buffer, [(4, 4), (5, 5), (3, 3)])
    
    def test_sample(self):
        exp_replay = RingBuffer(5)
        data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        for i in range(len(data)):
            exp_replay.append(data[i])
        sample1 = exp_replay.sample(3)
        sample2 = exp_replay.sample(3)
        self.assertNotEqual(sample1, sample2)

    def test_sample_len(self):
        exp_replay = RingBuffer(6)
        data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        for i in range(len(data)):
            exp_replay.append(data[i])
        sample1 = exp_replay.sample(3)
        self.assertEqual(len(sample1), 3)


if __name__=="__main__":
    unittest.main()