import os
import gym
import csv
import math
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from ..agents.DQNAgent import DQNAgent
from ..models.AtariCNN import AtariCNN, AtariCNN_bn
from ..common.utils.RingBuffer import RingBuffer
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

def make_net(inp_shape, num_actions, model_type="cnn_bn"):
    if model_type == "cnn_bn":
        PolicyNet = AtariCNN_bn(inp_shape, num_actions)
        TargetNet = AtariCNN_bn(inp_shape, num_actions)
    else:
        PolicyNet = AtariCNN(inp_shape, num_actions)
        TargetNet = AtariCNN(inp_shape, num_actions)        
    return PolicyNet, TargetNet


class DQNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.env = make_env(args.env_id)
        self.transition = args.transition
        self.replay_size = args.replay_size
        self.warm_start = args.warm_start
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.update_every = args.update_every
        self.device = args.device
        self.save_dir = args.save_dir
        self.writer = args.writer
        self.replay_memory = RingBuffer(self.replay_size)
        self.agent = DQNAgent(self.env, self.replay_memory, self.device)
        self.get_model(args)
        self.num_steps = 0
        self.total_reward = 0.0
        self.total_loss = 0.0
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # Fill the replay buffer with some elements for warm_start
        self.populate(steps=self.warm_start)
        print("Populated the replay buffer")
        self.get_optimizer(args)

        # Create the directory to save model files and results
        self.save_dir = os.path.join(self.save_dir, datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_model(self, args):
        if args.model == "CNN_bn":
            self.policy_net, self.target_net = make_net([4, 84, 84], self.env.action_space.n, "CNN_bn")
        else:
            self.policy_net, self.target_net = make_net([4, 84, 84], self.env.action_space.n, "CNN")
    
    def get_optimizer(self, args):
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        elif args.optimizer == "RMSProp":
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=args.lr, momentum=args.momentum)
    
    def populate(self, steps):
        for i in tqdm(range(steps)):
            self.agent.play_step(self.policy_net, epsilon=1.0)
    
    def sample_batch(self):
        sample = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = sample
        states, actions, rewards, next_states, dones = np.array(states), np.array(actions), np.array(rewards), \
            np.array(next_states), np.array(dones)
        states = torch.Tensor(states).type(torch.float32).to(self.device) / 255.0
        actions = torch.Tensor(actions).type(torch.LongTensor).to(self.device)
        rewards = torch.Tensor(rewards).type(torch.float32).to(self.device)
        next_states = torch.Tensor(next_states).type(torch.float32).to(self.device) / 255.0
        done = torch.Tensor(dones).type(torch.bool).to(self.device)
        return states, actions, rewards, next_states, done

    def update_epsilon(self, steps_done):
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * steps_done / self.epsilon_decay)
        return epsilon_threshold
    
    def optimize_model(self):
        states, actions, rewards, next_states, dones = self.sample_batch()
        state_action_vals = self.policy_net(states)
        state_action_vals = torch.gather(state_action_vals, 1, actions.unsqueeze(-1)).squeeze(-1)
        print("-"*100)
        print("dones.sum(): {}".format(dones.sum()))
        print(dones)
        with torch.no_grad():
            out = self.target_net(next_states)
            next_state_action_vals, idx = torch.max(out, dim=1)
            next_state_action_vals[dones] = 0.0
            next_state_action_vals = next_state_action_vals.detach()
        print(next_state_action_vals)
        target_state_action_vals = rewards + self.gamma * next_state_action_vals
        loss = F.smooth_l1_loss(state_action_vals, target_state_action_vals)
        
        # Take update step
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients bw -1 and 1
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        # Take optimizer step
        self.optimizer.step()
        if self.args.debug:
            # Compute the sum of L2 norm of all the gradients
            # to verify if gradients are exploding or not
            total_norm = 0.0
            for p in self.policy_net.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            return loss.item(), total_norm
        return loss.item(), None
    
    def save_current_state(self, episode):
        save_path = os.path.join(self.save_dir, "episode_{}.pth".format(episode))
        torch.save({
            'episode': episode,
            'policy_net_model_state_dict': self.policy_net.state_dict(),
            'target_net_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
    
    def write_save_logs(self, episode, loss, reward):
        rewards_path = os.path.join(self.save_dir, "rewards.txt")
        with open(rewards_path, "a") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([episode, reward])

        loss_path = os.path.join(self.save_dir, "loss.txt")
        with open(loss_path, "a") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([episode, loss])
        
        self.writer.add_scalar('training_loss_avg', loss, episode)
        self.writer.add_scalar('training_reward_avg', reward, episode)

    def train_loop(self, num_episodes):
        num_steps, best_reward = 0, -1e6
        episode_loss, episode_rewards = [], []
        global_start_time = time.time()
        for episode in range(num_episodes):
            ep_start_time = time.time()
            self.agent.reset()
            episode_len = 0.0
            while True:
                num_steps += 1
                episode_len += 1
                epsilon = self.update_epsilon(num_steps)
                reward, done = self.agent.play_step(self.policy_net, epsilon)
                self.total_reward += reward
                
                loss, grad = self.optimize_model()
                if self.args.debug:
                    self.writer.add_scalar('grad', grad, num_steps)
                self.total_loss += loss
                
                # Update the target network after every `update_every` steps
                if num_steps % self.update_every == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # If episode ends, reset the total reward and the state
                if done:
                    episode_rewards.append(self.total_reward)
                    episode_loss.append(self.total_loss / episode_len)
                    self.total_reward = 0.0
                    self.total_loss = 0.0
                    self.writer.add_scalar('episode_len', episode_len, episode)
                    break
            
            print("Episode Num: {}, Episode time: {}".format(episode, time.time() - ep_start_time))
            print("Episode Loss: {}, Episode Reward: {}".format(episode_loss[-1], episode_rewards[-1]))
            self.writer.add_scalar('training_loss', episode_loss[-1], episode)
            self.writer.add_scalar('training_reward', episode_rewards[-1], episode)
            # Remove magic number below
            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_loss = np.mean(np.mean(episode_loss[-100:]))
                print("-"*100)
                print("Num Episodes: {}, total steps: {}, epsilon: {}".format(episode, num_steps, epsilon))
                print("Average reward over last 100 episodes: {}".format(avg_reward))
                print("Average loss over last 100 episodes: {}".format(avg_loss))
                print("Total time taken: {}".format(time.time() - global_start_time))
                print("-"*100)
                start_time = time.time()
                if avg_reward > best_reward and episode > 100:
                    best_reward = avg_reward
                    self.save_current_state(episode)
                self.write_save_logs(episode, avg_reward, avg_reward)
