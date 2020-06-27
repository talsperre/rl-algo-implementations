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
from rl_models.common.utils.visualization import disp_frames
from rl_models.common.wrappers.AtariWrappers import wrap_deepmind, wrap_pytorch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

random.seed(42)

env = make_env('BreakoutDeterministic-v4')
is_done = False
state = env.reset()

while not is_done:
    state, reward, is_done, _ = env.step(env.action_space.sample())
    env.render()
    disp_frames(state)
