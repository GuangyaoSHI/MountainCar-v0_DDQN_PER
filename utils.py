import math
import pickle
import os

import torch
import torch.nn as nn

import numpy as np

def init_weights(m):
    if type(m) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        size = m.weight.data.size()
        size = size[1:] #without out_channels
        vol = np.prod(np.array(size))
        
        std = math.sqrt(2 / vol)

        torch.nn.init.normal(m.weight.data, mean=0, std=std)

        torch.nn.init.constant(m.bias.data, val=0)
    elif type(m) == nn.Linear:
        n = m.weight.data.size()[1] #only input channels

        std = math.sqrt(2 / n)

        torch.nn.init.normal(m.weight.data, mean=0, std=std)

        torch.nn.init.constant(m.bias.data, val=0)
    elif type(m) == nn.LSTM:
        size_weights = m.weight_ih_l0.data.size()[1]
        
        std = math.sqrt(2 / size_weights)
    
        for param in m.__dict__['_parameters'].keys():
            if param[:11] in ['weight_ih_l', 'weight_hh_l']:
                torch.nn.init.normal(m.__dict__['_parameters'][param].data, mean=0, std=std)
                torch.nn.init.normal(m.__dict__['_parameters'][param].data, mean=0, std=std)
            elif param[:9] in ['bias_ih_l', 'bias_hh_l']:
                torch.nn.init.constant(m.__dict__['_parameters'][param].data, val=1)
                torch.nn.init.constant(m.__dict__['_parameters'][param].data, val=1)
    else:
        print('Couldn\'t init wieghts of layer with type:', type(m))

def find_prev_state_files(models_dir, state_file):
    timestamps = [int(name) for name in os.listdir(models_dir)]
    in_dir = os.path.join(models_dir, str(max(timestamps)))

    timestamps = [int(name) for name in os.listdir(in_dir) if name not in [state_file, state_file + '_bak']]
    timestamps.sort()

    weights_file_path = os.path.join(in_dir, str(timestamps[-1]))
    state_file_path = os.path.join(in_dir, state_file)

    return weights_file_path, state_file_path

def dirac_delta(i, n):
    ret = [0 for i in range(n)]
    ret[i] = 1
    return ret