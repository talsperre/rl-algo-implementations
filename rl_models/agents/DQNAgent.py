import torch
import numpy as np
np.random.seed(42)


class DQNAgent(object):
    def __init__(self, env, replay_memory, device):
        self.env = env
        self.replay_memory = replay_memory
        self.device = device        
        self.reset()
    
    def reset(self):
        self.state = self.env.reset()

    def get_state_inp(self, state):
        state = torch.Tensor(state).type(torch.float32) / 255.0
        state = state.unsqueeze(dim=0).to(self.device)
        return state
    
    def play_step(self, net, epsilon):
        action = self.select_action(net, epsilon)
        next_state, reward, is_done, _ = self.env.step(action)
        self.replay_memory.append((self.state, action, reward, next_state, is_done))
        self.state = next_state
        if is_done:
            self.reset()
        return reward, is_done

    def select_action(self, net, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # Choose the best action based on Q-Value
            with torch.no_grad():
                # Need to normalize inputs to range of 0-1
                inp = self.get_state_inp(self.state)
                out = net(inp)
                _, idx = torch.max(out, dim=1)
                action = int(idx.item())
        return action