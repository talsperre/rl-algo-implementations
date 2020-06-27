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

env = gym.make('PongDeterministic-v4')
env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
env = wrap_pytorch(env)
env.seed(42)

# lst = []
# for i in range(1):
#     state = env.reset()
#     done = False
#     tot_reward = 0.0
#     while not done:
#         state, reward, done, _ = env.step(env.action_space.sample())
#         env.render()
#         time.sleep(0.01)
#         tot_reward += reward
#     print(i, tot_reward, state.shape)
#     lst.append(tot_reward)
# print(np.mean(lst))
# disp_frames(state)

env = make_env('PongDeterministic-v4')
PolicyNet, TargetNet = make_net([4, 84, 84], env.action_space.n)
replay_memory = RingBuffer(10000)
agent = DQNAgent(env, replay_memory, device)
lst = []
for i in range(100):
    is_done = False
    tot_reward = 0.0
    while not is_done:
        reward, is_done = agent.play_step(PolicyNet, epsilon=0.5)
        tot_reward += reward
        print(agent.replay_memory.buffer[-1][4])
    print(agent.replay_memory.buffer[-1][4])
    print(is_done)
    break
#     print(i, tot_reward)
#     lst.append(tot_reward)
# print(np.mean(lst))