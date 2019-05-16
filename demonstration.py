import torch
from torch.autograd import Variable

from QNet import QNet
from env import Env
import utils

import numpy as np

import cv2

import os
import random
import time

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

k = 4

def myzscore(x):
    mean = np.mean(x)
    std = np.std(x)

    ret = x - mean
    if std != 0:
        ret = ret / std

    return ret

def prepare_state(screen_arr):
    ret = [cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) for screen in screen_arr]

    ret = [myzscore(screen.astype(np.float32)) for screen in ret]

    ret = [np.expand_dims(screen, axis=0) for screen in ret]
    ret = np.concatenate(ret, axis=0)

    ret = np.expand_dims(ret, axis=0)

    return ret

if __name__ == '__main__':
    #Engine init
    env = Env({
        'border_width' : 50,
        'square_size' : 20,
        'snake_color' : np.array([90, 0, 157]),
        'food_color' : np.array([255, 0, 0]),
        'background_color' : np.array([255, 255, 255]),
        'border_background_color' : np.array([255, 255, 255]),
        'border_line_color' : np.array([0, 0, 0]),
        'snake_head_color' : np.array([0, 255, 0]),
        'border_line_width' : 3,
        'field_size' : (8, 8),
        'snake_init_len' : 4,
        'food_count' : 1,
        'food_score' : 1,
        'death_score' : -1,
        'survive_score' : 0,
        'torus' : False
    })

    #Load names
    models_dir = 'models'
    state_file = 'state'

    #Loading model
    Q_targ = QNet(4, 4)
    weights_file_path, _ = utils.find_prev_state_files(models_dir, state_file)
    weights = utils.load_file(weights_file_path)
    Q_targ.set_weights(weights['Q_targ'])

    #Demonstration
    while True:
        env.new_game()

        hashes = set()

        last_frames = [np.full((10, 10, 3), fill_value=255, dtype=np.uint8) for i in range(k - 1)]
        last_frames.append(env.screenshot())

        while not env.finished():
            env.render()
            Env.draw()

            curr_state = prepare_state(last_frames)
            
            action = None

            screen_hash = hash(curr_state.tostring())
            if screen_hash in hashes:
                print('Entered loop...')
                
                action = random.randint(0, 3)
            hashes.add(screen_hash)

            if action is None:
                action = Variable(FloatTensor(curr_state), volatile=True)
                action = Q_targ(action)
                action = action.cpu().data.numpy()[0]
                #print(action)
                action = np.argmax(action)

            env.act(action)

            last_frames = last_frames[1:] + [env.screenshot()]

            time.sleep(0.01)

        print('Score:', env.score)

        time.sleep(0.3)