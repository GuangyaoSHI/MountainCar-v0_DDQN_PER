import time
import random
import pickle
import os
import sys
import math

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np

from QNet import QNet
from replay_memory import ReplayMemory
from env import Env
import utils

import cv2

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

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
    #Save/load dirs
    data_dir = './data'
    models_dir = './models'
    state_file = 'state'
    out_dir = os.path.join(models_dir, str(int(time.time())))

    #Engine parameters
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

    num_actions = 4

    #Last frames count definition
    k = 4

    #Initing neural networks and loading previos state, if exists
    Q = QNet(num_actions=num_actions, k=k)
    Q_targ = QNet(num_actions=num_actions, k=k)

    prev_state = None
    if os.listdir(models_dir) == []:
        Q.apply(utils.init_weights)

        Q_targ.set_weights(Q.get_weights())
    else:
        weights_file_path, state_file_path = utils.find_prev_state_files(models_dir, state_file)

        weights = utils.load_file(weights_file_path)
        prev_state = utils.load_file(state_file_path)

        Q.set_weights(weights['Q'])
        Q_targ.set_weights(weights['Q_targ'])

    #Learn params
    gamma = 0.99

    #Hyperparams
    frame_count = 7008000 #219 target updates
    eps_decay_time = 0.5

    eps_start = 1
    eps_end = 0.05
    l = -math.log(eps_end) / (frame_count * eps_decay_time)

    replay_mem = ReplayMemory(max_size=500000, alpha=0.5, eps=0.0) if prev_state is None else prev_state['replay_mem']

    episode_depth = 10000

    batch_size = 32

    Q_targ_update_freq = 32000

    save_freq = frame_count / 100

    #Episode loop
    curr_eps = eps_start if prev_state is None else prev_state['end_eps']

    episode_num = 0 if prev_state is None else (prev_state['end_episode'] + 1)
    curr_frame_count = 0 if prev_state is None else prev_state['curr_frame_count']

    #Optimizer init
    optimizer = torch.optim.Adam(Q.parameters(), lr=0.0000625, eps=1.5e-4)
    
    while True:
        if curr_frame_count > frame_count:
            break

        print()
        print('===================================================')
        print('Episode number:', episode_num)
        print('Frames processed:', round(curr_frame_count / frame_count * 100, 2), '%')

        losses = []

        env.new_game()

        last_frames = [np.full((10, 10, 3), fill_value=255, dtype=np.uint8) for i in range(k - 1)]
        last_frames.append(env.screenshot())

        curr_state = prepare_state(last_frames)
        Q_out = Q(Variable(FloatTensor(curr_state), volatile=1)).data.cpu().numpy()[0]

        for t in range(episode_depth):
            #
            # Acting
            #

            if env.finished():
                break

            print(t, ' ', end='')
            sys.stdout.flush()

            action = None
            if random.random() < curr_eps:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q_out)

            reward = env.act(action)

            loss = reward - np.amax(Q_out)

            new_state = None
            if not env.finished():
                last_frames = last_frames[1:] + [env.screenshot()]
                new_state = prepare_state(last_frames)

                Q_out = Q(Variable(FloatTensor(new_state), volatile=1)).data.cpu().numpy()[0]
                Q_targ_out = Q_targ(Variable(FloatTensor(new_state), volatile=1)).data.cpu().numpy()[0]

                loss += gamma * Q_targ_out[np.argmax(Q_out)]

            loss = abs(loss)

            replay_mem.add_element((curr_state, action, reward, new_state), loss)

            curr_state = new_state

            #
            #Learning
            #

            sarses = replay_mem.get_batch(batch_size)

            #Targets

            Q_true = []
            if any([sars[3] is not None for sars in sarses]):
                batch = np.concatenate([sars[3] for sars in sarses if sars[3] is not None], axis=0)
                batch = Variable(FloatTensor(batch), volatile=1)

                actions = np.argmax(Q(batch).data.cpu().numpy(), axis=1)
                Q_targ_out = Q_targ(batch).data.cpu().numpy()
                Q_true = [Q_targ_out[i][actions[i]] for i in range(Q_targ_out.shape[0])]
            
            j = 0
            Q_true_new = []
            for i in range(len(sarses)):
                if sarses[i][3] is None:
                    Q_true_new.append(0)
                else:
                    Q_true_new.append(Q_true[j])
                    j += 1
            Q_true = np.array(Q_true_new)

            rs = np.array([sars[2] for sars in sarses])
            Q_true = rs + gamma * Q_true

            #Predictions

            batch = np.concatenate([sars[0] for sars in sarses], axis=0)
            batch = Variable(FloatTensor(batch), requires_grad=True)
            Q_pred = Q(batch)
            mask = np.array([utils.dirac_delta(i=sars[1], n=num_actions) for sars in sarses])
            mask = Variable(ByteTensor(mask), requires_grad=False)
            Q_pred = torch.masked_select(Q_pred, mask=mask)

            #Updating replay memory

            replay_mem.update(np.abs(Q_true - Q_pred.data.cpu().numpy()))

            #Learning

            loss = nn.SmoothL1Loss()(Q_pred, Variable(FloatTensor(Q_true), requires_grad=False))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.data.cpu().numpy()[0]))

            #
            #Updating frame counter
            #

            curr_frame_count += 1

            #
            #Updating eps
            #

            curr_eps = max(math.exp(-l * curr_frame_count), eps_end)

            #
            #Updating Q_targ if needed
            #

            if curr_frame_count % Q_targ_update_freq == 0:
                print('Updating Q_targ')

                Q_targ.set_weights(Q.get_weights())

            #
            #Saving if needed
            #

            if curr_frame_count % save_freq == 0:
                print('Saving model')

                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                fd = open(os.path.join(out_dir, str(curr_frame_count)), 'wb')
                pickle.dump({'Q': Q.get_weights(), 'Q_targ' : Q_targ.get_weights()}, fd)
                fd.close()

                state_file_path = os.path.join(out_dir, state_file)

                if os.path.exists(state_file_path):
                    bak_state_file_path = os.path.join(out_dir, state_file + '_bak')

                    if os.path.exists(bak_state_file_path):
                        os.remove(bak_state_file_path)
                        os.rename(state_file_path, bak_state_file_path)

                fd = open(state_file_path, 'wb')
                pickle.dump({'replay_mem' : replay_mem,
                    'end_eps' : curr_eps,
                    'end_episode' : episode_num,
                    'curr_frame_count' : curr_frame_count
                }, fd)
                fd.close()

        #
        #Other info
        #

        print()
        print('Average loss:', np.mean(losses))
        print('Replay memory size:', len(replay_mem))
        print('Eps:', curr_eps)

        #
        #Episode loop routine
        #

        episode_num += 1

    print()
    print('Done!')