import torch

from QNet import QNet
import utils

import numpy as np

import os
import random
import time

import gym

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def prepare_state(state):
    ret = np.array([(state[0]+.3)/.9, state[1]/.07])
    ret = np.expand_dims(ret, axis=0)
    return ret

if __name__ == '__main__':
    #Engine init
    env = gym.make('MountainCar-v0')

    #Load names
    models_dir = 'models'
    state_file = 'state'

    #Loading model
    Q_targ = QNet(3)
    weights_file_path, _ = utils.find_prev_state_files(models_dir, state_file)
    weights = torch.load(weights_file_path)
    Q_targ.load_state_dict(weights['Q_targ'])

    #Demonstration
    while True:
        env.reset()
        env.render()

        raw_state, reward, done, info = env.step(1)

        while not done:
            state = prepare_state(raw_state)

            action = np.argmax(Q_targ(FloatTensor(state)).to('cpu').detach().numpy())

            raw_state, reward, done, info = env.step(action)
            env.render()

            time.sleep(0.01)

        time.sleep(0.3)