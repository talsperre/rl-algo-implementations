import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AtariCNN(nn.Module):
    def __init__(self, inp_shape, num_actions):
        super(AtariCNN, self).__init__()
        # Conv Layers to downsample image
        self.features = nn.Sequential(
            nn.Conv2d(inp_shape[0], 32, 8, 4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(self._get_conv_out(inp_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_out(self, shape):
        # Returns the size of the tensor after flattening output
        out = self.features(torch.zeros(1, *shape))
        return int(np.prod(out.size()))
    
    def forward(self, x):
        _out = self.features(x)
        # Flatten outputs - Change dims to batch_size x N
        _out = _out.view(_out.size(0), -1)
        _out = self.head(_out)
        return _out