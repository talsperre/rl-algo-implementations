import sys
sys.path.append("../")
import gym
import time
import random
import unittest
import rl_models

from rl_models.common.wrappers.AtariWrappers import wrap_deepmind, wrap_pytorch
from rl_models.common.utils.visualization import disp_frames
random.seed(42)


class TestAtariEnv(unittest.TestCase):
    def test_frame_shape(self):
        env = gym.make('BreakoutDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        observation = env.reset()
        self.assertEqual(observation.shape, (4, 84, 84))
    
    def test_num_actions(self):
        env = gym.make('BreakoutDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        observation = env.reset()
        self.assertEqual(env.action_space.n, 4)

    def test_pong_num_actions(self):
        env = gym.make('PongDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        observation = env.reset()
        self.assertEqual(env.action_space.n, 6)
    
    def test_pong_rewards(self):
        env = gym.make('PongDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        
        state = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            state, reward, done, _ = env.step(env.action_space.sample())
            tot_reward += reward
        self.assertGreaterEqual(tot_reward, -21)
        self.assertLessEqual(tot_reward, 21)
    
    def test_breakout_rewards(self):
        env = gym.make('BreakoutDeterministic-v4')
        env = wrap_deepmind(env, frame_stack=True)
        env = wrap_pytorch(env)
        env.seed(42)
        
        state = env.reset()
        done = False
        tot_reward = 0.0
        while not done:
            state, reward, done, _ = env.step(env.action_space.sample())
            tot_reward += reward
        self.assertGreaterEqual(tot_reward, 0)

if __name__=="__main__":
    unittest.main()