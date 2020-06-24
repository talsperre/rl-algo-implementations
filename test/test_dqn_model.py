import sys
sys.path.append("../")
import gym
import time
import torch
import random
import unittest
import rl_models
import numpy as np

from rl_models.models.AtariCNN import AtariCNN
from rl_models.common.wrappers.AtariWrappers import wrap_deepmind, wrap_pytorch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class TestDQNModel(unittest.TestCase):
    def test_output_shape(self):
        inps = torch.zeros(10, 4, 84, 84)
        DQNModel = AtariCNN(inps.size()[1:], num_actions=4)
        out = DQNModel(inps)
        self.assertEqual(out.size(), (10, 4))
    
    def test_env_input(self):
        env = gym.make('BreakoutDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        state = env.reset()
        # Need to normalize inputs to range of 0-1
        state = torch.tensor(state, dtype=torch.float32) / 255.0 
        state = state.unsqueeze(dim=0)
        DQNModel = AtariCNN(env.observation_space.shape, num_actions=4)
        out = DQNModel(state)
        self.assertEqual(out.size(), (1, 4))

if __name__=="__main__":
    unittest.main()