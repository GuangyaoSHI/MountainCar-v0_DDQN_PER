import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pickle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class QNet(nn.Module):
    def __init__(self, num_actions):
        super(QNet, self).__init__()  

        self.lin1 = nn.Linear(in_features=2, out_features=40)
        self.lin2 = nn.Linear(in_features=40, out_features=num_actions)

        global use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x