import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pickle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class QNet(nn.Module):
    def __init__(self, num_actions, k):
        super(QNet, self).__init__()  

        self.conv1 = nn.Conv2d(in_channels=k, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        self.lin1 = nn.Linear(in_features=1152, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=num_actions)

        global use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x) 

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self, path):
        theta = [np.copy(x.data.cpu().numpy()) for x in self.parameters()]

        fd = open(path, 'wb')
        pickle.dump(theta, fd)
        fd.close()

    def load(self, path):
        fd = open(path, 'rb')
        theta = pickle.load(fd)
        fd.close()

        for dest, src in zip(self.parameters(), theta):
            dest.data.copy_(FloatTensor(src))

    def get_weights(self):
        return [np.copy(x.data.cpu().numpy()) for x in self.parameters()]

    def set_weights(self, theta):
        i = 0

        for p in self.parameters():
            p.data.copy_(FloatTensor(theta[i]))
            
            i += 1