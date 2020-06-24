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
from rl_models.agents.DQNAgent import DQNAgent
from rl_models.common.utils.RingBuffer import RingBuffer
from rl_models.common.wrappers.AtariWrappers import wrap_deepmind, wrap_pytorch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def make_env(env_id):
    env = gym.make(env_id)
    env = wrap_deepmind(env, frame_stack=True)
    env = wrap_pytorch(env)
    env.seed(42)
    return env

def make_net(inp_shape, num_actions):
    PolicyNet = AtariCNN(inp_shape, num_actions)
    TargetNet = AtariCNN(inp_shape, num_actions)
    return PolicyNet, TargetNet


class TestDQNAgent(unittest.TestCase):
    def test_random_action_type(self):
        env = make_env("BreakoutDeterministic-v4")
        PolicyNet, TargetNet = make_net([4, 84, 84], env.action_space.n)
        replay_memory = RingBuffer(256)
        agent = DQNAgent(env, replay_memory)
        action = agent.select_action(PolicyNet, epsilon=1)
        self.assertIsInstance(action, int)

    def test_greedy_action_type(self):
        env = make_env("BreakoutDeterministic-v4")
        PolicyNet, TargetNet = make_net([4, 84, 84], env.action_space.n)
        replay_memory = RingBuffer(256)
        agent = DQNAgent(env, replay_memory)
        action = agent.select_action(PolicyNet, epsilon=0.0)
        self.assertIsInstance(action, int)
    
    def test_play_step(self):
        env = make_env("BreakoutDeterministic-v4")
        PolicyNet, TargetNet = make_net([4, 84, 84], env.action_space.n)
        replay_memory = RingBuffer(256)
        agent = DQNAgent(env, replay_memory)
        reward, is_done = agent.play_step(PolicyNet, epsilon=0.0)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(is_done, bool)
        self.assertEqual(len(agent.replay_memory), 1)

if __name__=="__main__":
    unittest.main()