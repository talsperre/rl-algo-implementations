import os
import sys
sys.path.append("../")
import gym
import time
import torch
import random
import unittest
import argparse
import rl_models
import numpy as np

from collections import namedtuple
from rl_models.models.AtariCNN import AtariCNN
from rl_models.train.DQNTrain import DQNTrainer
from torch.utils.tensorboard import SummaryWriter
from rl_models.common.wrappers.AtariWrappers import wrap_deepmind, wrap_pytorch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--env_id", help="Environment id to train on", default="BreakoutDeterministic-v4")
parser.add_argument("--model", help="Model", default="CNN_bn")
parser.add_argument("--debug", help="print debug statements", default=False, type=bool)
parser.add_argument("--optimizer", help="optimizer type", default="Adam")
parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
parser.add_argument("--momentum", help="momentum", default=0.9, type=float)
parser.add_argument("--beta1", help="beta1", default=0.9, type=float)
parser.add_argument("--beta2", help="beta2", default=0.999, type=float)
parser.add_argument("--replay_size", help="Size of replay buffer", default=100000, type=int)
parser.add_argument("--warm_start", help="Num steps before sampling", default=50000, type=int)
parser.add_argument("--batch_size", help="batch size", default=32, type=int)
parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
parser.add_argument("--epsilon_start", help="initial epsilon value", default=1.0, type=float)
parser.add_argument("--epsilon_end", help="minimum epsilon value", default=0.01, type=float)
parser.add_argument("--epsilon_decay", help="decay factor of epsilon", default=3e4, type=int)
parser.add_argument("--update_every", help="Num steps before updating target weights", default=1000, type=int)
parser.add_argument("--save_dir", help="Directory where models are saved", default="../cache")
args = parser.parse_args()

# default `log_dir` is "runs" - we'll be more specific here
os.makedirs(os.path.join(args.save_dir, 'runs'), exist_ok=True)
writer = SummaryWriter(os.path.join(args.save_dir, 'runs', 'breakout_experiment2'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
args.device = device
args.transition = Transition
args.writer = writer

dqn_trainer = DQNTrainer(args)
# dqn_trainer.train_loop(num_episodes=1)
dqn_trainer.train_loop(num_episodes=10000)
